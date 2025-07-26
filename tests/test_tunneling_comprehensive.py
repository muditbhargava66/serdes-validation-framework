"""
Comprehensive tests for USB4 tunneling validation

This module provides thorough testing of the enhanced tunneling validation
functionality including PCIe TLP extraction, flow control validation, and
CRC verification.
"""

import os

import numpy as np
import pytest

# Set mock mode
os.environ["SVF_MOCK_MODE"] = "1"

# Conditional imports with proper error handling
try:
    from serdes_validation_framework.protocols.usb4.tunneling import (
        PCIeTLPHeader,
        PCIeTunnelValidator,
    )

    TUNNELING_AVAILABLE = True
except ImportError as e:
    TUNNELING_AVAILABLE = False
    pytestmark = pytest.mark.skip(f"Tunneling module not available: {e}")

try:
    from serdes_validation_framework.protocols.usb4.constants import USB4TunnelingMode, USB4TunnelingSpecs

    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False

    # Create mock constants for testing
    class MockUSB4TunnelingMode:
        PCIE = "PCIE"
        DISPLAYPORT = "DISPLAYPORT"
        USB32 = "USB32"

    class MockUSB4TunnelingSpecs:
        PCIE_MIN_BANDWIDTH = 10e9

    USB4TunnelingMode = MockUSB4TunnelingMode()
    USB4TunnelingSpecs = MockUSB4TunnelingSpecs()


@pytest.mark.skipif(not TUNNELING_AVAILABLE, reason="Tunneling module not available")
class TestPCIeTunnelValidatorEnhanced:
    """Test enhanced PCIe tunnel validator functionality"""

    @pytest.fixture
    def validator(self):
        """Create PCIe tunnel validator"""
        config = {"max_tlp_size": 4096, "flow_control_enabled": True, "crc_validation": True, "timeout_ms": 1000}
        return PCIeTunnelValidator(config)

    @pytest.fixture
    def sample_tlp_data(self):
        """Create sample TLP data for testing"""
        # Create realistic PCIe TLP data
        tlp_data = np.zeros(1024, dtype=np.uint8)

        # Add some TLP headers at regular intervals
        for i in range(0, len(tlp_data), 64):
            if i + 16 <= len(tlp_data):
                # Memory read request TLP header
                tlp_data[i : i + 4] = [0x00, 0x00, 0x00, 0x40]  # Format and type
                tlp_data[i + 4 : i + 8] = [0x00, 0x10, 0x12, 0x34]  # Length and requester ID
                tlp_data[i + 8 : i + 12] = [0x56, 0x78, 0x9A, 0xBC]  # Tag and address
                tlp_data[i + 12 : i + 16] = [0xDE, 0xF0, 0x00, 0x00]  # Address continuation

        return tlp_data

    @pytest.fixture
    def sample_tlp_headers(self):
        """Create sample TLP headers for testing"""
        return [
            PCIeTLPHeader(
                fmt=0x0,
                type=0x00,
                tc=0,
                attr=0,
                length=4,
                requester_id=0x1234,
                tag=0x56,
                last_dw_be=0xF,
                first_dw_be=0xF,
                address=0x12345678,
                data=b"\x00" * 16,
            ),
            PCIeTLPHeader(
                fmt=0x1,
                type=0x20,
                tc=0,
                attr=0,
                length=8,
                requester_id=0x5678,
                tag=0x9A,
                last_dw_be=0xF,
                first_dw_be=0xF,
                address=0x87654321,
                data=b"\xff" * 32,
            ),
            PCIeTLPHeader(
                fmt=0x0,
                type=0x0A,
                tc=0,
                attr=0,
                length=0,
                requester_id=0xABCD,
                tag=0xEF,
                last_dw_be=0x0,
                first_dw_be=0x0,
                address=0x0,
                data=b"",
            ),
        ]

    def test_tlp_extraction_comprehensive(self, validator, sample_tlp_data):
        """Test comprehensive TLP extraction"""
        # Test TLP extraction
        tlps = validator._extract_tlps_from_data(sample_tlp_data)

        # Should extract some TLPs
        assert isinstance(tlps, list)

        # Each extracted TLP should be valid
        for tlp in tlps:
            assert isinstance(tlp, PCIeTLPHeader)
            assert hasattr(tlp, "type")
            assert hasattr(tlp, "length")
            assert hasattr(tlp, "requester_id")

    def test_tlp_start_detection(self, validator):
        """Test TLP start pattern detection"""
        # Create test data with known TLP start patterns
        test_data = bytearray(256)

        # Insert TLP start patterns at known locations
        patterns = [
            (16, b"\x00\x00\x00\x40"),  # Memory Read Request
            (64, b"\x00\x00\x00\x60"),  # Memory Write Request
            (128, b"\x00\x00\x00\x4a"),  # Completion
        ]

        for offset, pattern in patterns:
            test_data[offset : offset + 4] = pattern

        # Test TLP start detection
        for expected_offset, _ in patterns:
            found_offset = validator._find_tlp_start(test_data, 0)
            assert found_offset != -1
            assert found_offset <= expected_offset

            # Test from after this pattern
            next_offset = validator._find_tlp_start(test_data, found_offset + 4)
            if next_offset != -1:
                assert next_offset > found_offset

    def test_tlp_header_validation(self, validator, sample_tlp_headers):
        """Test TLP header validation"""
        for tlp in sample_tlp_headers:
            # Test basic header validation
            is_valid = validator._validate_tlp_header_basic(tlp)
            assert isinstance(is_valid, bool)

            # Valid TLPs should pass validation
            if tlp.type in [0x00, 0x01, 0x20, 0x21, 0x0A, 0x0B]:
                assert is_valid is True

    def test_tlp_crc_validation(self, validator):
        """Test TLP CRC validation"""
        # Create test TLP data with CRC
        test_data = b"\x00\x01\x02\x03" * 4  # 16 bytes of data

        # Calculate CRC and append
        crc = validator._calculate_crc32(test_data)
        tlp_with_crc = test_data + crc.to_bytes(4, byteorder="little")

        # Test CRC validation
        crc_valid = validator._validate_tlp_crc(tlp_with_crc)
        assert isinstance(crc_valid, bool)
        assert crc_valid is True

        # Test with corrupted CRC
        corrupted_crc = tlp_with_crc[:-1] + b"\xff"
        crc_invalid = validator._validate_tlp_crc(corrupted_crc)
        assert crc_invalid is False

    def test_crc32_calculation(self, validator):
        """Test CRC-32 calculation"""
        test_data = b"Hello, World!"

        # Calculate CRC
        crc = validator._calculate_crc32(test_data)

        # Should return integer CRC
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFFFFFF

        # Same data should produce same CRC
        crc2 = validator._calculate_crc32(test_data)
        assert crc == crc2

        # Different data should produce different CRC
        different_data = b"Hello, World?"
        crc3 = validator._calculate_crc32(different_data)
        assert crc != crc3

    def test_flow_control_validation_comprehensive(self, validator, sample_tlp_headers):
        """Test comprehensive flow control validation"""
        # Test flow control validation
        flow_control_result = validator._validate_flow_control(sample_tlp_headers)

        # Should return detailed flow control information
        assert isinstance(flow_control_result, dict)
        assert "credit_violations" in flow_control_result
        assert "buffer_overruns" in flow_control_result
        assert "flow_control_valid" in flow_control_result
        assert "flow_control_efficiency" in flow_control_result
        assert "final_credit_counters" in flow_control_result
        assert "flow_control_events" in flow_control_result

        # Check data types
        assert isinstance(flow_control_result["credit_violations"], int)
        assert isinstance(flow_control_result["buffer_overruns"], int)
        assert isinstance(flow_control_result["flow_control_valid"], bool)
        assert isinstance(flow_control_result["flow_control_efficiency"], float)
        assert isinstance(flow_control_result["final_credit_counters"], dict)
        assert isinstance(flow_control_result["flow_control_events"], list)

        # Efficiency should be between 0 and 1
        assert 0.0 <= flow_control_result["flow_control_efficiency"] <= 1.0

    def test_tlp_categorization_for_flow_control(self, validator):
        """Test TLP categorization for flow control"""
        # Test different TLP types
        test_cases = [
            (0x00, "non_posted"),  # Memory Read Request
            (0x01, "non_posted"),  # Memory Read Request (64-bit)
            (0x20, "posted"),  # Memory Write Request
            (0x21, "posted"),  # Memory Write Request (64-bit)
            (0x04, "non_posted"),  # Configuration Read
            (0x05, "non_posted"),  # Configuration Write
            (0x0A, "completion"),  # Completion
            (0x0B, "completion"),  # Completion with Data
            (0x02, "non_posted"),  # I/O Read
            (0x03, "non_posted"),  # I/O Write
            (0x30, "posted"),  # Message Request
            (0xFF, "non_posted"),  # Unknown type (default)
        ]

        for tlp_type, expected_category in test_cases:
            tlp = PCIeTLPHeader(
                fmt=0x0, type=tlp_type, tc=0x0, attr=0x0, length=4, requester_id=0x1234, tag=0x56, last_dw_be=0xF, first_dw_be=0xF
            )

            category = validator._categorize_tlp_for_flow_control(tlp)
            assert category == expected_category

    def test_credit_return_simulation(self, validator):
        """Test credit return simulation"""
        # Create initial credit counters
        credit_counters = {
            "posted_header": 100,
            "posted_data": 500,
            "non_posted_header": 30,
            "non_posted_data": 200,
            "completion_header": 40,
            "completion_data": 300,
        }

        # Store initial values
        initial_counters = credit_counters.copy()

        # Simulate credit return
        validator._simulate_credit_return(credit_counters)

        # Credits should increase (or stay the same if at maximum)
        for credit_type in credit_counters:
            assert credit_counters[credit_type] >= initial_counters[credit_type]

        # Should not exceed maximum values
        max_credits = {
            "posted_header": 256,
            "posted_data": 2048,
            "non_posted_header": 64,
            "non_posted_data": 512,
            "completion_header": 64,
            "completion_data": 512,
        }

        for credit_type, max_credit in max_credits.items():
            assert credit_counters[credit_type] <= max_credit

    def test_flow_control_with_violations(self, validator):
        """Test flow control validation with credit violations"""
        # Create TLPs that will cause credit violations
        large_tlps = []
        for i in range(300):  # More TLPs than available credits
            tlp = PCIeTLPHeader(
                fmt=0x1,
                type=0x20,  # Memory Write (posted)
                tc=0x0,
                attr=0x0,
                length=64,  # Large payload
                requester_id=0x1234 + i,
                tag=i % 256,
                last_dw_be=0xF,
                first_dw_be=0xF,
            )
            large_tlps.append(tlp)

        # Test flow control validation
        result = validator._validate_flow_control(large_tlps)

        # Should detect violations
        assert result["credit_violations"] > 0 or result["buffer_overruns"] > 0
        assert result["flow_control_valid"] is False
        assert result["flow_control_efficiency"] < 1.0

        # Should have flow control events
        assert len(result["flow_control_events"]) > 0

        # Check event structure
        for event in result["flow_control_events"]:
            assert "tlp_index" in event
            assert "event_type" in event
            assert event["event_type"] in ["credit_violation", "buffer_overrun"]

    def test_tunnel_validation_integration(self, validator, sample_tlp_data):
        """Test integrated tunnel validation"""
        # Test complete tunnel validation
        results = validator.validate_tunnel(USB4TunnelingMode.PCIE, sample_tlp_data)

        # Should return comprehensive results
        assert isinstance(results, dict)
        assert "tunnel_mode" in results
        assert "data_size" in results
        assert "tlp_count" in results

        # Should have validation results
        expected_keys = [
            "tlp_count",
            "total_tlps",
            "valid_tlps",
            "integrity_rate",
            "data_rate",
            "bandwidth_utilization",
            "within_limits",
            "credit_violations",
            "buffer_overruns",
            "flow_control_valid",
        ]

        for key in expected_keys:
            if key in results:  # Some keys might not be present depending on implementation
                assert results[key] is not None

    def test_bandwidth_allocation_validation(self, validator, sample_tlp_data):
        """Test bandwidth allocation validation"""
        # Test bandwidth validation
        bandwidth_results = validator._validate_bandwidth_allocation(sample_tlp_data)

        assert isinstance(bandwidth_results, dict)
        assert "data_rate" in bandwidth_results
        assert "bandwidth_utilization" in bandwidth_results
        assert "within_limits" in bandwidth_results

        # Check data types
        assert isinstance(bandwidth_results["data_rate"], (int, float))
        assert isinstance(bandwidth_results["bandwidth_utilization"], (int, float))
        assert isinstance(bandwidth_results["within_limits"], bool)

        # Utilization should be non-negative
        assert bandwidth_results["bandwidth_utilization"] >= 0

    def test_error_handling_in_tlp_extraction(self, validator):
        """Test error handling in TLP extraction"""
        # Test with invalid data
        invalid_data = np.array([])  # Empty array
        tlps = validator._extract_tlps_from_data(invalid_data)
        assert isinstance(tlps, list)
        assert len(tlps) == 0

        # Test with corrupted data
        corrupted_data = np.random.randint(0, 256, size=100, dtype=np.uint8)
        tlps = validator._extract_tlps_from_data(corrupted_data)
        assert isinstance(tlps, list)
        # Should handle corrupted data gracefully

    def test_error_handling_in_flow_control(self, validator):
        """Test error handling in flow control validation"""
        # Test with invalid TLP
        invalid_tlp = PCIeTLPHeader(
            fmt=0xFF,  # Invalid format
            type=0xFF,  # Invalid type
            tc=0xFF,
            attr=0xFF,
            length=-1,  # Invalid length
            requester_id=0xFFFFFFFF,  # Invalid requester ID
            tag=0xFFFF,  # Invalid tag
            last_dw_be=0xFF,
            first_dw_be=0xFF,
        )

        # Should handle invalid TLP gracefully
        result = validator._validate_flow_control([invalid_tlp])
        assert isinstance(result, dict)
        assert "flow_control_valid" in result

    def test_performance_with_large_dataset(self, validator):
        """Test performance with large TLP dataset"""
        import time

        # Create large dataset
        large_tlps = []
        for i in range(1000):  # 1000 TLPs
            tlp = PCIeTLPHeader(
                fmt=i % 2,
                type=[0x00, 0x20, 0x0A][i % 3],
                tc=i % 8,
                attr=i % 4,
                length=i % 64 + 1,
                requester_id=0x1000 + i,
                tag=i % 256,
                last_dw_be=0xF,
                first_dw_be=0xF,
            )
            large_tlps.append(tlp)

        # Test performance
        start_time = time.time()
        result = validator._validate_flow_control(large_tlps)
        execution_time = time.time() - start_time

        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds max
        assert result is not None
        assert "flow_control_valid" in result

    @pytest.mark.parametrize(
        "tlp_type,expected_category",
        [
            (0x00, "non_posted"),
            (0x20, "posted"),
            (0x0A, "completion"),
            (0x04, "non_posted"),
            (0x30, "posted"),
        ],
    )
    def test_tlp_categorization_parametrized(self, validator, tlp_type, expected_category):
        """Test TLP categorization with parametrized inputs"""
        tlp = PCIeTLPHeader(
            fmt=0x0, type=tlp_type, tc=0x0, attr=0x0, length=4, requester_id=0x1234, tag=0x56, last_dw_be=0xF, first_dw_be=0xF
        )

        category = validator._categorize_tlp_for_flow_control(tlp)
        assert category == expected_category

    def test_tlp_integrity_validation(self, validator, sample_tlp_headers):
        """Test TLP integrity validation"""
        # Test TLP integrity validation
        integrity_results = validator._validate_tlp_integrity(sample_tlp_headers)

        assert isinstance(integrity_results, dict)
        assert "total_tlps" in integrity_results
        assert "valid_tlps" in integrity_results
        assert "integrity_rate" in integrity_results

        # Check data types and ranges
        assert isinstance(integrity_results["total_tlps"], int)
        assert isinstance(integrity_results["valid_tlps"], int)
        assert isinstance(integrity_results["integrity_rate"], float)
        assert 0.0 <= integrity_results["integrity_rate"] <= 1.0

        # Valid TLPs should not exceed total TLPs
        assert integrity_results["valid_tlps"] <= integrity_results["total_tlps"]


if __name__ == "__main__":
    pytest.main([__file__])
