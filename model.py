from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class _Model(torch.nn.Module):

    """
    Base class for all models
    """

    def __init__(self):
        super(_Model, self).__init__()

    @classmethod
    def from_config(cls, config: dict):
        """ All models should have this class method """
        raise NotImplementedError
class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )

        # shape (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        return stft_f
class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)
def index2one_hot(index_tensor, vocabulary_size):
    """
    Transforms index representation to one hot representation
    :param index_tensor: shape: (batch_size, sequence_length, 1) tensor containing character indices
    :param vocabulary_size: scalar, size of the vocabulary
    :return: chars_one_hot: shape: (batch_size, sequence_length, vocabulary_size)
    """

    device = index_tensor.device
    index_tensor = index_tensor.type(torch.LongTensor).to(device)

    batch_size = index_tensor.size()[0]
    char_sequence_len = index_tensor.size()[1]
    chars_one_hot = torch.zeros((batch_size, char_sequence_len, vocabulary_size), device=device)
    chars_one_hot.scatter_(dim=2, index=index_tensor, value=1)

    return chars_one_hot

class OpenUnmix(_Model):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        scaler_mean = config['scaler_mean'] if 'scaler_mean' in keys else None
        scaler_std = config['scaler_std'] if 'scaler_std' in keys else None
        return cls(input_mean=scaler_mean,
                   input_scale=scaler_std,
                   nb_channels=config['nb_channels'],
                   hidden_size=config['hidden_size'],
                   n_fft=config['nfft'],
                   n_hop=config['nhop'],
                   max_bin=config['max_bin'],
                   sample_rate=config['samplerate']
                   )

    def forward(self, x):

        # ignore potential side info that has been given as input
        x = x[0]

        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x
class InformedOpenUnmix3(_Model):
    """
    Open Unmix with an additional text encoder and attention mechanism
    """
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=16000,
        audio_encoder_layers=2,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
        vocab_size=44,
        attention='general',
        audio_transform = 'STFT'
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(InformedOpenUnmix3, self).__init__()

        self.return_alphas = False
        self.optimal_path_alphas = False

        # text processing
        self.vocab_size = vocab_size
        self.attention = attention

        self.lstm_txt = LSTM(vocab_size, hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)

        # attention

        w_s_init = torch.empty(hidden_size, hidden_size)
        k = torch.sqrt(torch.tensor(1).type(torch.float32) / hidden_size)
        nn.init.uniform_(w_s_init, -k, k)
        self.w_s = nn.Parameter(w_s_init, requires_grad=True)

        # connection
        self.fc_c = Linear(hidden_size * 2, hidden_size)
        self.bn_c = BatchNorm1d(hidden_size)

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        elif audio_transform == 'STFT':
            self.transform = nn.Sequential(self.stft, self.spec)

        # audio encoder
        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.audio_encoder_lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size,
                                       num_layers=audio_encoder_layers, bidirectional=not unidirectional,
                                       batch_first=False, dropout=0.4)


        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        scaler_mean = config['scaler_mean'] if 'scaler_mean' in keys else None
        scaler_std = config['scaler_std'] if 'scaler_std' in keys else None
        attention = config['attention'] if 'attention' in keys else 'general'
        return cls(input_mean=scaler_mean,
                   input_scale=scaler_std,
                   nb_channels=config['nb_channels'],
                   hidden_size=config['hidden_size'],
                   n_fft=config['nfft'],
                   n_hop=config['nhop'],
                   max_bin=config['max_bin'],
                   sample_rate=config['samplerate'],
                   vocab_size=config['vocabulary_size'],
                   audio_encoder_layers=config['nb_audio_encoder_layers'],
                   attention=attention)

    def forward(self, x):

        text_idx = x[1].unsqueeze(dim=2)  # text as index sequence
        x = x[0]  # mix

        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # -------------------------------------------------------------------------------------------------------------
        # text processing
        text_onehot = index2one_hot(text_idx, self.vocab_size)  # shape (nb_samples, sequence_len, vocabulary_size)

        h, _ = self.lstm_txt(text_onehot)  # lstm expects shape (batch_size, sequence_len, nb_features)

        # -------------------------------------------------------------------------------------------------------------
        # audio processing

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        x, _ = self.audio_encoder_lstm(x)

        # -------------------------------------------------------------------------------------------------------------
        # attention
        batch_size = h.size(0)
        x = x.transpose(0, 1)  # to shape (nb_samples, nb_frames, self.hidden_size)

        # compute score = g_n * W_s * h_m in two steps
        side_info_transformed = torch.bmm(self.w_s.expand(batch_size, -1, -1),
                                          torch.transpose(h, 1, 2))

        scores = torch.bmm(x, side_info_transformed)
        self.attention = "dtw"

        if self.attention == 'general':
            # compute the attention weights of all side information steps for all audio time steps
            alphas = F.softmax(scores, dim=2)  # shape: (nb_samples, N, M)
        elif self.attention == 'dtw':
            if self.optimal_path_alphas:
                # use the (non-differentiable) optimal path as attention weights (at test time if desired)
                alphas = torch.tensor(optimal_alignment_path(scores), device=scores.device)\
                    .unsqueeze(0).to(torch.float32)
            else:
                dtw_alphas = dtw_matrix(scores, mode='faster')
                alphas = F.softmax(dtw_alphas, dim=2)


        # compute context vectors
        context = torch.bmm(torch.transpose(h, 1, 2), torch.transpose(alphas, 1, 2))

        # make shape: (nb_samples, N, hidden_size)
        context = torch.transpose(context, 1, 2)

        # -------------------------------------------------------------------------------------------------------------
        # connection of audio and text
        concat = torch.cat((context, x), dim=2)
        x = self.fc_c(concat)
        x = self.bn_c(x.transpose(1, 2))  # (nb_samples, hidden_size, nb_frames)
        x = torch.tanh(x)

        x = x.transpose(1, 2)
        x = x.transpose(0, 1)  # --> (nb_frames, nb_samples, hidden_size)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        if self.return_alphas:
            if self.attention == 'general':
                return x, alphas
            elif self.attention == 'dtw':
                return x, alphas, scores

        return x
class InformedOpenUnmix3NA2(_Model):
    """
    like InformedOpenUnmix3 but no attention mechanism. Alignment is given to the model as input
    """
    def __init__(
            self,
            n_fft=4096,
            n_hop=1024,
            input_is_spectrogram=False,
            hidden_size=512,
            nb_channels=2,
            sample_rate=44100,
            audio_encoder_layers=2,
            nb_layers=3,
            input_mean=None,
            input_scale=None,
            max_bin=None,
            unidirectional=False,
            power=1,
            vocab_size=32
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(InformedOpenUnmix3NA2, self).__init__()


        # text processing
        self.vocab_size = vocab_size

        self.lstm_txt = LSTM(vocab_size, hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)
        self.drop_out = nn.Dropout(0.4)

        # connection
        self.fc_c = Linear(hidden_size * 2, hidden_size)
        self.bn_c = BatchNorm1d(hidden_size)

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)


        # audio encoder
        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.audio_encoder_lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size,
                                       num_layers=audio_encoder_layers, bidirectional=not unidirectional,
                                       batch_first=False, dropout=0.4)

        # mask decoder
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        scaler_mean = config['scaler_mean'] if 'scaler_mean' in keys else None
        scaler_std = config['scaler_std'] if 'scaler_std' in keys else None
        return cls(input_mean=scaler_mean,
                   input_scale=scaler_std,
                   nb_channels=config['nb_channels'],
                   hidden_size=config['hidden_size'],
                   n_fft=config['nfft'],
                   n_hop=config['nhop'],
                   max_bin=config['max_bin'],
                   sample_rate=config['samplerate'],
                   vocab_size=config['vocabulary_size'],
                   audio_encoder_layers=config['nb_audio_encoder_layers'],
                   )

    def forward(self, x):

        attention_weights = x[2]
        text_idx = x[1].unsqueeze(dim=2)  # text as index sequence
        x = x[0]  # mix

        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # -------------------------------------------------------------------------------------------------------------
        # text processing
        text_onehot = index2one_hot(text_idx, self.vocab_size)  # shape (nb_samples, sequence_len, vocabulary_size)

        h, _ = self.lstm_txt(text_onehot)  # lstm expects shape (batch_size, sequence_len, nb_features)
        h = self.drop_out(h)

        # -------------------------------------------------------------------------------------------------------------
        # audio processing

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        x, _ = self.audio_encoder_lstm(x)

        # -------------------------------------------------------------------------------------------------------------
        # alignment of text with audio
        context = torch.bmm(torch.transpose(h, 1, 2), torch.transpose(attention_weights, 1, 2))

        # make shape: (nb_samples, N, hidden_size)
        context = torch.transpose(context, 1, 2)

        # -------------------------------------------------------------------------------------------------------------
        # connection of audio and text
        x = x.transpose(0, 1)  # to shape (nb_samples, nb_frames, self.hidden_size)
        concat = torch.cat((context, x), dim=2)
        x = self.fc_c(concat)
        x = self.bn_c(x.transpose(1, 2))  # (nb_samples, hidden_size, nb_frames)
        x = torch.tanh(x)

        x = x.transpose(1, 2)
        x = x.transpose(0, 1)  # --> (nb_frames, nb_samples, hidden_size)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x