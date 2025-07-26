#!/usr/bin/env python3
"""
Test Runner for SerDes Validation Framework

This script provides a comprehensive test runner for all test suites in the framework.
It supports different test categories and provides detailed reporting.

Usage:
    python tests/run_tests.py [options]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src and tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set mock mode for testing
os.environ["SVF_MOCK_MODE"] = "1"

# Test categories and their corresponding files
TEST_CATEGORIES = {
    "unit": ["test_basic_functionality.py", "test_summary.py"],
    "comprehensive": [
        "test_data_analysis.py",
        "test_framework_integration.py",
        "test_pcie_analyzer.py",
        "test_usb4_comprehensive.py",
        "test_multi_protocol.py",
    ],
    "integration": ["integration/test_multi_protocol_integration.py", "integration/test_usb4_integration.py"],
    "performance": ["performance/test_usb4_performance_regression.py"],
    "legacy": [
        "legacy/test_data_collection.py",
        "legacy/test_instrument_control.py",
        "legacy/test_test_sequence.py",
        "legacy/test_dual_mode.py",
        "legacy/test_eth_224g_sequence.py",
        "legacy/test_nrz_analyzer.py",
        "legacy/test_pam4_analyzer.py",
        "legacy/test_pcie_integration.py",
        "legacy/test_pcie_sequence.py",
        "legacy/test_scope_224g.py",
        "legacy/test_usb4_eye_diagram.py",
        "legacy/test_usb4_jitter_analyzer.py",
        "legacy/test_usb4_link_recovery.py",
        "legacy/test_usb4_link_training.py",
        "legacy/test_usb4_power_management.py",
        "legacy/test_usb4_signal_analyzer.py",
        "legacy/test_usb4_tunneling.py",
    ],
}


class TestRunner:
    """Test runner for SerDes Validation Framework"""

    def __init__(self, test_dir: Path):
        """Initialize test runner"""
        self.test_dir = test_dir
        self.results = {}

    def run_pytest(self, test_files: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run pytest on specified test files"""
        if not test_files:
            return {"status": "skipped", "reason": "No test files specified"}

        # Filter existing test files
        existing_files = []
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                existing_files.append(str(test_path))
            else:
                print(f"Warning: Test file {test_file} not found")

        if not existing_files:
            return {"status": "skipped", "reason": "No existing test files found"}

        # Build pytest command
        cmd = ["python", "-m", "pytest"] + existing_files
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "--no-header"])

        try:
            # Set up environment for pytest
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.test_dir.parent)
            env["SVF_MOCK_MODE"] = "1"

            # Run pytest
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent, env=env)

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "files_tested": len(existing_files),
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "files_tested": 0}

    def run_category(self, category: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category"""
        if category not in TEST_CATEGORIES:
            return {"status": "error", "error": f"Unknown category: {category}"}

        print(f"\n{'='*60}")
        print(f"Running {category.upper()} tests...")
        print(f"{'='*60}")

        test_files = TEST_CATEGORIES[category]
        result = self.run_pytest(test_files, verbose)

        # Print results
        if result["status"] == "passed":
            print(f"✅ {category.upper()} tests PASSED ({result['files_tested']} files)")
        elif result["status"] == "failed":
            print(f"❌ {category.upper()} tests FAILED ({result['files_tested']} files)")
            if verbose and result.get("stdout"):
                print("\nOutput:")
                print(result["stdout"])
            if result.get("stderr"):
                print("\nErrors:")
                print(result["stderr"])
        elif result["status"] == "skipped":
            print(f"⏭️  {category.upper()} tests SKIPPED - {result['reason']}")
        else:
            print(f"💥 {category.upper()} tests ERROR - {result.get('error', 'Unknown error')}")

        return result

    def run_all_categories(self, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run all test categories"""
        results = {}

        print("🚀 Starting SerDes Validation Framework Test Suite")
        print(f"Mock mode: {'ENABLED' if os.environ.get('SVF_MOCK_MODE') == '1' else 'DISABLED'}")

        for category in ["unit", "comprehensive", "integration", "performance"]:
            results[category] = self.run_category(category, verbose)

        return results

    def run_legacy_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run legacy tests (may have import issues)"""
        print(f"\n{'='*60}")
        print("Running LEGACY tests (may have import issues)...")
        print(f"{'='*60}")

        return self.run_category("legacy", verbose)

    def print_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print test summary"""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        total_categories = len(results)
        passed_categories = sum(1 for r in results.values() if r.get("status") == "passed")
        failed_categories = sum(1 for r in results.values() if r.get("status") == "failed")
        skipped_categories = sum(1 for r in results.values() if r.get("status") == "skipped")
        error_categories = sum(1 for r in results.values() if r.get("status") == "error")

        print(f"Categories tested: {total_categories}")
        print(f"✅ Passed: {passed_categories}")
        print(f"❌ Failed: {failed_categories}")
        print(f"⏭️  Skipped: {skipped_categories}")
        print(f"💥 Errors: {error_categories}")

        # Detailed results
        for category, result in results.items():
            status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "error": "💥"}.get(result.get("status"), "❓")

            files_tested = result.get("files_tested", 0)
            print(f"{status_icon} {category.upper()}: {result.get('status', 'unknown').upper()} ({files_tested} files)")

        # Overall status
        if failed_categories == 0 and error_categories == 0:
            print("\n🎉 Overall: SUCCESS")
            return True
        else:
            print("\n⚠️  Overall: ISSUES DETECTED")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SerDes Validation Framework Test Runner")
    parser.add_argument(
        "--category",
        choices=["unit", "comprehensive", "integration", "performance", "legacy", "all"],
        default="all",
        help="Test category to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--include-legacy", action="store_true", help="Include legacy tests (may have import issues)")

    args = parser.parse_args()

    # Initialize test runner
    test_dir = Path(__file__).parent
    runner = TestRunner(test_dir)

    try:
        if args.category == "all":
            # Run main test categories
            results = runner.run_all_categories(args.verbose)

            # Run legacy tests if requested
            if args.include_legacy:
                results["legacy"] = runner.run_legacy_tests(args.verbose)

            # Print summary
            success = runner.print_summary(results)
            sys.exit(0 if success else 1)

        elif args.category == "legacy":
            # Run only legacy tests
            result = runner.run_legacy_tests(args.verbose)
            success = result.get("status") == "passed"
            sys.exit(0 if success else 1)

        else:
            # Run specific category
            result = runner.run_category(args.category, args.verbose)
            success = result.get("status") == "passed"
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
