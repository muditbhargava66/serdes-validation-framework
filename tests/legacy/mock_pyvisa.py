from unittest.mock import MagicMock

mock_rm = MagicMock()
mock_instrument = MagicMock()
mock_rm.open_resource.return_value = mock_instrument


def mock_pyvisa():
    return mock_rm, mock_instrument
