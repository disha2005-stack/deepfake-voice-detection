"""
lstm_model.py
-------------
Bidirectional LSTM (BiLSTM) classifier for Mel-spectrogram images.

Input representation
--------------------
Each Mel-spectrogram is treated as a time-frequency matrix:
  - Rows (H) = frequency bins   -> feature dimension F
  - Columns (W) = time frames   -> sequence length T

After loading and resizing:
  image tensor: [C=1, H=freq, W=time]
  Squeeze channel, permute -> [T, F]
  Batch -> [batch, T, F]  (batch_first=True in LSTM)

Architecture
------------
  Input  : [batch, T=128, F=64]
  BiLSTM : 2 layers, hidden=128, dropout=0.3 between layers
           num_directions=2, so total hidden dim = 256

  Classification head:
    hidden tensor shape from nn.LSTM: [num_layers * num_directions, batch, hidden]
                                    = [4, batch, 128]
    hidden[-2] -> final forward  hidden state of last layer: [batch, 128]
    hidden[-1] -> final backward hidden state of last layer: [batch, 128]
    Concatenate -> [batch, 256]
    Dropout(0.4) -> FC(256, 2)

  Why hidden[-2] and hidden[-1] and NOT output[:, -1, :]:
    - output[:, -1, :] gives the LAST TIMESTEP's concatenated forward+backward
      hidden state. For the backward direction this is actually the FIRST input
      timestep's state, not the final recurrent state. This is a subtle but
      real bug when taken literally as "the final state."
    - The `hidden` return from nn.LSTM always contains the TRUE final recurrent
      state for each layer and each direction, regardless of sequence length.
    - hidden[-2, :, :] is the FORWARD  direction's final state (last layer).
    - hidden[-1, :, :] is the BACKWARD direction's final state (last layer).
    This is the standard and correct approach for BiLSTM classification.

Labels : real=0, fake=1  (positive class for evaluation = FAKE = 1)
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size: int = 64,      # F: number of frequency bins per time step
        hidden_size: int = 128,    # LSTM hidden units per direction
        num_layers: int = 2,       # stacked LSTM layers
        dropout: float = 0.3,      # dropout between LSTM layers (not applied on last layer)
        fc_dropout: float = 0.4,   # dropout before the final FC layer
        num_classes: int = 2,      # REAL=0, FAKE=1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM.
        # dropout is applied BETWEEN layers (not after the last layer).
        # When num_layers=1 PyTorch ignores the dropout argument; here
        # num_layers=2 so it is active.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # expects [batch, seq, feature]
            bidirectional=True,
            dropout=dropout,
        )

        # Classification head.
        # hidden[-2] and hidden[-1] each have shape [batch, hidden_size]
        # -> concatenated: [batch, 2 * hidden_size]
        self.fc_dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [batch, T, F]
            T = time steps (sequence length), F = frequency bins (features)

        Returns
        -------
        logits : torch.Tensor  shape [batch, num_classes]
        """
        # output shape : [batch, T, 2 * hidden_size]
        # hidden shape : [num_layers * 2, batch, hidden_size]
        # cell  shape  : [num_layers * 2, batch, hidden_size]
        _output, (hidden, _cell) = self.lstm(x)

        # Extract true final states for each direction (last BiLSTM layer only).
        # hidden[-2] = forward  direction, last layer -> shape [batch, hidden_size]
        # hidden[-1] = backward direction, last layer -> shape [batch, hidden_size]
        forward_final  = hidden[-2]   # [batch, hidden_size]
        backward_final = hidden[-1]   # [batch, hidden_size]

        # Concatenate -> [batch, 2 * hidden_size]
        combined = torch.cat([forward_final, backward_final], dim=1)

        # Apply dropout then linear projection.
        out = self.fc_dropout(combined)
        logits = self.fc(out)         # [batch, num_classes]

        return logits


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly to verify dimensions)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("BiLSTMClassifier -- dimension self-test")
    print("=" * 50)

    model = BiLSTMClassifier(
        input_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        fc_dropout=0.4,
        num_classes=2,
    )
    model.eval()

    batch = 4
    T = 128   # time steps
    F = 64    # frequency bins

    x = torch.randn(batch, T, F)
    with torch.no_grad():
        logits = model(x)

    print(f"Input  shape : {list(x.shape)}   expected [{batch}, {T}, {F}]")
    print(f"Output shape : {list(logits.shape)}      expected [{batch}, 2]")
    assert logits.shape == torch.Size([batch, 2]), "Shape mismatch!"
    print("Shape check  : PASSED OK")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Approx size (FP32)  : {total_params * 4 / 1024**2:.2f} MB")

    # Show hidden state shapes explicitly
    x2 = torch.randn(2, T, F)
    _output, (hidden, _cell) = model.lstm(x2)
    print(f"\nLSTM hidden shape   : {list(hidden.shape)}   (num_layers*2, batch, hidden)")
    print(f"hidden[-2] (fwd)    : {list(hidden[-2].shape)}")
    print(f"hidden[-1] (bwd)    : {list(hidden[-1].shape)}")
    concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
    print(f"Concatenated        : {list(concat.shape)}")
    print("\nAll checks PASSED OK")
    sys.exit(0)
