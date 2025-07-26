#!/usr/bin/env python3
"""
USB4 Link Recovery Testing Demo

This example demonstrates the USB4 link recovery testing capabilities,
including error injection, recovery timing validation, error logging and analysis,
and recovery mechanism effectiveness measurement.

Features demonstrated:
- Error injection and recovery testing methods
- Link recovery timing validation
- Comprehensive error logging and analysis
- Recovery mechanism effectiveness measurement
- Statistical analysis of recovery performance
- Detailed recovery diagnostics and reporting
"""

import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serdes_validation_framework.protocols.usb4.constants import (
    USB4ErrorType,
    USB4SignalMode,
)
from serdes_validation_framework.protocols.usb4.link_recovery import (
    USB4ErrorInjectionConfig,
    USB4ErrorSeverity,
    USB4LinkRecoveryTester,
    USB4RecoveryConfig,
    USB4RecoveryStatus,
)


def demonstrate_basic_error_recovery():
    """Demonstrate basic error injection and recovery"""
    print("=" * 60)
    print("USB4 Basic Error Recovery Testing Demo")
    print("=" * 60)
    
    # Create recovery configuration
    config = USB4RecoveryConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=50.0e9,
        capture_length=1000000,
        max_recovery_time=5.0,
        recovery_timeout=10.0,
        max_recovery_attempts=3,
        enable_statistics=True,
        enable_detailed_logging=True,
        measure_timing=True,
        validate_signal_quality=True
    )
    
    # Initialize recovery tester
    print("Initializing USB4 link recovery tester...")
    tester = USB4LinkRecoveryTester(config)
    
    if not tester.initialize():
        print("Failed to initialize recovery tester")
        return
    
    try:
        # Test different error types
        error_types = [
            USB4ErrorType.LINK_TRAINING,
            USB4ErrorType.SIGNAL_INTEGRITY,
            USB4ErrorType.PROTOCOL,
            USB4ErrorType.POWER_MANAGEMENT
        ]
        
        for error_type in error_types:
            print(f"\n--- Testing {error_type.name} Error Recovery ---")
            
            # Configure error injection
            injection_config = USB4ErrorInjectionConfig(
                error_type=error_type,
                severity=USB4ErrorSeverity.MEDIUM,
                duration=0.1,
                recovery_timeout=5.0,
                enable_logging=True
            )
            
            # Perform error injection and recovery test
            results = tester.test_error_injection_and_recovery(injection_config)
            
            # Display results
            print(f"Test Duration: {results.test_duration:.3f}s")
            print(f"Recovery Attempts: {len(results.recovery_attempts)}")
            print(f"Test Result: {'PASSED' if results.test_passed else 'FAILED'}")
            
            # Show recovery attempt details
            for i, attempt in enumerate(results.recovery_attempts, 1):
                print(f"  Attempt {i}: {attempt.method.name} -> {attempt.status.name} ({attempt.duration:.3f}s)")
            
            # Show effectiveness metrics
            if results.effectiveness_metrics:
                print(f"Success Rate: {results.effectiveness_metrics.get('success_rate', 0):.3f}")
                if 'average_quality_improvement' in results.effectiveness_metrics:
                    print(f"Quality Improvement: {results.effectiveness_metrics['average_quality_improvement']:.3f}")
    
    finally:
        tester.cleanup()
    
    print("\nBasic error recovery testing completed!")


def demonstrate_comprehensive_recovery_testing():
    """Demonstrate comprehensive recovery testing"""
    print("\n" + "=" * 60)
    print("USB4 Comprehensive Recovery Testing Demo")
    print("=" * 60)
    
    # Create comprehensive recovery configuration
    config = USB4RecoveryConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=50.0e9,
        capture_length=1000000,
        max_recovery_time=3.0,
        recovery_timeout=15.0,
        max_recovery_attempts=5,
        enable_statistics=True,
        enable_detailed_logging=True,
        test_all_error_types=True,
        measure_timing=True,
        validate_signal_quality=True,
        stress_test_duration=2.0,  # Short duration for demo
        error_injection_rate=2.0   # 2 errors per second
    )
    
    # Initialize recovery tester
    print("Initializing comprehensive recovery tester...")
    tester = USB4LinkRecoveryTester(config)
    
    if not tester.initialize():
        print("Failed to initialize recovery tester")
        return
    
    try:
        # Perform comprehensive recovery testing
        print("Starting comprehensive recovery testing...")
        start_time = time.time()
        
        results = tester.test_comprehensive_recovery()
        
        test_duration = time.time() - start_time
        print(f"Comprehensive testing completed in {test_duration:.3f}s")
        
        # Display comprehensive results
        print("\n--- Comprehensive Test Results ---")
        print(f"Total Errors Tested: {results.total_errors_tested}")
        print(f"Total Recovery Attempts: {len(results.recovery_attempts)}")
        print(f"Overall Test Result: {'PASSED' if results.test_passed else 'FAILED'}")
        
        # Display recovery statistics
        stats = results.recovery_statistics
        print("\n--- Recovery Statistics ---")
        print(f"Success Rate: {stats.recovery_success_rate:.3f}")
        print(f"Average Recovery Time: {stats.average_recovery_time:.3f}s")
        print(f"Min Recovery Time: {stats.min_recovery_time:.3f}s")
        print(f"Max Recovery Time: {stats.max_recovery_time:.3f}s")
        print(f"Successful Recoveries: {stats.successful_recoveries}")
        print(f"Failed Recoveries: {stats.failed_recoveries}")
        print(f"Timeout Recoveries: {stats.timeout_recoveries}")
        
        # Display timing validation results
        if results.timing_validation_results:
            timing = results.timing_validation_results
            print("\n--- Timing Validation ---")
            print(f"Timing Compliance: {'PASS' if timing.get('timing_compliance', False) else 'FAIL'}")
            print(f"Timing Violations: {timing.get('timing_violations', 0)}")
            print(f"Violation Rate: {timing.get('timing_violation_rate', 0):.3f}")
        
        # Display effectiveness metrics
        if results.effectiveness_metrics:
            effectiveness = results.effectiveness_metrics
            print("\n--- Effectiveness Metrics ---")
            print(f"Overall Success Rate: {effectiveness.get('overall_success_rate', 0):.3f}")
            print(f"First Attempt Success Rate: {effectiveness.get('first_attempt_success_rate', 0):.3f}")
            
            if 'average_quality_improvement' in effectiveness:
                print(f"Average Quality Improvement: {effectiveness['average_quality_improvement']:.3f}")
            if 'average_bandwidth_preservation' in effectiveness:
                print(f"Average Bandwidth Preservation: {effectiveness['average_bandwidth_preservation']:.3f}")
            if 'power_efficient_recovery_rate' in effectiveness:
                print(f"Power Efficient Recovery Rate: {effectiveness['power_efficient_recovery_rate']:.3f}")
        
        # Display recommendations
        if results.recommendations:
            print("\n--- Recommendations ---")
            for i, recommendation in enumerate(results.recommendations, 1):
                print(f"{i}. {recommendation}")
        
        # Display error type distribution
        error_types = {}
        for event in results.error_events:
            error_types[event.error_type] = error_types.get(event.error_type, 0) + 1
        
        print("\n--- Error Type Distribution ---")
        for error_type, count in error_types.items():
            print(f"{error_type.name}: {count}")
        
        # Display recovery method effectiveness
        method_stats = {}
        for attempt in results.recovery_attempts:
            if attempt.method not in method_stats:
                method_stats[attempt.method] = {'total': 0, 'success': 0}
            method_stats[attempt.method]['total'] += 1
            if attempt.status == USB4RecoveryStatus.SUCCESS:
                method_stats[attempt.method]['success'] += 1
        
        print("\n--- Recovery Method Effectiveness ---")
        for method, stats in method_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{method.name}: {success_rate:.3f} ({stats['success']}/{stats['total']})")
    
    finally:
        tester.cleanup()
    
    print("\nComprehensive recovery testing completed!")


def demonstrate_recovery_timing_analysis():
    """Demonstrate recovery timing analysis"""
    print("\n" + "=" * 60)
    print("USB4 Recovery Timing Analysis Demo")
    print("=" * 60)
    
    # Create timing-focused configuration
    config = USB4RecoveryConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=50.0e9,
        capture_length=1000000,
        max_recovery_time=2.0,  # Strict timing requirement
        recovery_timeout=10.0,
        max_recovery_attempts=3,
        enable_statistics=True,
        measure_timing=True,
        validate_signal_quality=True
    )
    
    # Initialize recovery tester
    print("Initializing recovery tester for timing analysis...")
    tester = USB4LinkRecoveryTester(config)
    
    if not tester.initialize():
        print("Failed to initialize recovery tester")
        return
    
    try:
        # Test multiple recovery scenarios for timing analysis
        all_recovery_attempts = []
        
        print("Performing multiple recovery tests for timing analysis...")
        
        for i in range(5):  # Perform 5 tests
            print(f"Test {i+1}/5...")
            
            # Alternate between different error types
            error_type = [USB4ErrorType.LINK_TRAINING, USB4ErrorType.SIGNAL_INTEGRITY][i % 2]
            
            injection_config = USB4ErrorInjectionConfig(
                error_type=error_type,
                severity=USB4ErrorSeverity.MEDIUM,
                duration=0.05,
                recovery_timeout=5.0
            )
            
            results = tester.test_error_injection_and_recovery(injection_config)
            all_recovery_attempts.extend(results.recovery_attempts)
        
        # Perform comprehensive timing validation
        print("\nAnalyzing recovery timing...")
        timing_results = tester.validate_recovery_timing(all_recovery_attempts)
        
        # Display timing analysis results
        print("\n--- Recovery Timing Analysis ---")
        print(f"Total Recovery Attempts: {len(all_recovery_attempts)}")
        print(f"Average Recovery Time: {timing_results['average_recovery_time']:.3f}s")
        print(f"Min Recovery Time: {timing_results['min_recovery_time']:.3f}s")
        print(f"Max Recovery Time: {timing_results['max_recovery_time']:.3f}s")
        print(f"Standard Deviation: {timing_results['std_recovery_time']:.3f}s")
        print(f"Max Allowed Time: {timing_results['max_allowed_time']:.3f}s")
        print(f"Timing Compliance: {'PASS' if timing_results['timing_compliance'] else 'FAIL'}")
        print(f"Timing Violations: {timing_results['timing_violations']}")
        print(f"Violation Rate: {timing_results['timing_violation_rate']:.3f}")
        
        # Display timing by error type
        if 'timing_by_error_type' in timing_results:
            print("\n--- Timing by Error Type ---")
            for error_type, stats in timing_results['timing_by_error_type'].items():
                print(f"{error_type}:")
                print(f"  Average: {stats['average']:.3f}s")
                print(f"  Max: {stats['max']:.3f}s")
                print(f"  Count: {stats['count']}")
        
        # Display timing by recovery method
        if 'timing_by_method' in timing_results:
            print("\n--- Timing by Recovery Method ---")
            for method, stats in timing_results['timing_by_method'].items():
                print(f"{method}:")
                print(f"  Average: {stats['average']:.3f}s")
                print(f"  Max: {stats['max']:.3f}s")
                print(f"  Count: {stats['count']}")
    
    finally:
        tester.cleanup()
    
    print("\nRecovery timing analysis completed!")


def demonstrate_effectiveness_measurement():
    """Demonstrate recovery effectiveness measurement"""
    print("\n" + "=" * 60)
    print("USB4 Recovery Effectiveness Measurement Demo")
    print("=" * 60)
    
    # Create effectiveness-focused configuration
    config = USB4RecoveryConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=50.0e9,
        capture_length=1000000,
        max_recovery_time=5.0,
        recovery_timeout=15.0,
        max_recovery_attempts=4,
        enable_statistics=True,
        validate_signal_quality=True,
        measure_timing=True
    )
    
    # Initialize recovery tester
    print("Initializing recovery tester for effectiveness measurement...")
    tester = USB4LinkRecoveryTester(config)
    
    if not tester.initialize():
        print("Failed to initialize recovery tester")
        return
    
    try:
        # Test different error severities for effectiveness analysis
        all_recovery_attempts = []
        
        print("Testing different error severities...")
        
        for severity in USB4ErrorSeverity:
            print(f"Testing {severity.name} severity errors...")
            
            for error_type in [USB4ErrorType.LINK_TRAINING, USB4ErrorType.SIGNAL_INTEGRITY]:
                injection_config = USB4ErrorInjectionConfig(
                    error_type=error_type,
                    severity=severity,
                    duration=0.1,
                    recovery_timeout=10.0
                )
                
                results = tester.test_error_injection_and_recovery(injection_config)
                all_recovery_attempts.extend(results.recovery_attempts)
        
        # Measure recovery effectiveness
        print("\nMeasuring recovery effectiveness...")
        effectiveness_metrics = tester.measure_recovery_effectiveness(all_recovery_attempts)
        
        # Display effectiveness measurement results
        print("\n--- Recovery Effectiveness Measurement ---")
        print(f"Total Recovery Attempts: {len(all_recovery_attempts)}")
        print(f"Overall Success Rate: {effectiveness_metrics['overall_success_rate']:.3f}")
        print(f"First Attempt Success Rate: {effectiveness_metrics['first_attempt_success_rate']:.3f}")
        
        # Display success rates by error type
        if 'success_by_error_type' in effectiveness_metrics:
            print("\n--- Success Rate by Error Type ---")
            for error_type, rate in effectiveness_metrics['success_by_error_type'].items():
                print(f"{error_type}: {rate:.3f}")
        
        # Display success rates by recovery method
        if 'success_by_method' in effectiveness_metrics:
            print("\n--- Success Rate by Recovery Method ---")
            for method, rate in effectiveness_metrics['success_by_method'].items():
                print(f"{method}: {rate:.3f}")
        
        # Display quality and performance metrics
        if 'average_quality_improvement' in effectiveness_metrics:
            print("\n--- Quality and Performance Metrics ---")
            print(f"Average Quality Improvement: {effectiveness_metrics['average_quality_improvement']:.3f}")
            print(f"Max Quality Improvement: {effectiveness_metrics['max_quality_improvement']:.3f}")
        
        if 'average_bandwidth_preservation' in effectiveness_metrics:
            print(f"Average Bandwidth Preservation: {effectiveness_metrics['average_bandwidth_preservation']:.3f}")
            print(f"Min Bandwidth Preservation: {effectiveness_metrics['min_bandwidth_preservation']:.3f}")
        
        if 'power_efficient_recovery_rate' in effectiveness_metrics:
            print(f"Power Efficient Recovery Rate: {effectiveness_metrics['power_efficient_recovery_rate']:.3f}")
        
        # Analyze recovery patterns
        print("\n--- Recovery Pattern Analysis ---")
        
        # Count attempts by status
        status_counts = {}
        for attempt in all_recovery_attempts:
            status_counts[attempt.status] = status_counts.get(attempt.status, 0) + 1
        
        print("Recovery Status Distribution:")
        for status, count in status_counts.items():
            percentage = (count / len(all_recovery_attempts)) * 100
            print(f"  {status.name}: {count} ({percentage:.1f}%)")
        
        # Analyze multi-attempt recoveries
        multi_attempt_errors = {}
        for attempt in all_recovery_attempts:
            if attempt.attempt_number > 1:
                error_type = attempt.error_type
                multi_attempt_errors[error_type] = multi_attempt_errors.get(error_type, 0) + 1
        
        if multi_attempt_errors:
            print("\nMulti-Attempt Recovery Analysis:")
            for error_type, count in multi_attempt_errors.items():
                print(f"  {error_type.name}: {count} multi-attempt recoveries")
    
    finally:
        tester.cleanup()
    
    print("\nRecovery effectiveness measurement completed!")


def demonstrate_recovery_report_generation():
    """Demonstrate recovery analysis report generation"""
    print("\n" + "=" * 60)
    print("USB4 Recovery Report Generation Demo")
    print("=" * 60)
    
    # Create configuration for report generation
    config = USB4RecoveryConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=50.0e9,
        capture_length=1000000,
        max_recovery_time=3.0,
        recovery_timeout=10.0,
        max_recovery_attempts=3,
        enable_statistics=True,
        enable_detailed_logging=True,
        test_all_error_types=True,
        stress_test_duration=1.0,
        error_injection_rate=1.0
    )
    
    # Initialize recovery tester
    print("Initializing recovery tester for report generation...")
    tester = USB4LinkRecoveryTester(config)
    
    if not tester.initialize():
        print("Failed to initialize recovery tester")
        return
    
    try:
        # Perform comprehensive testing for report
        print("Performing comprehensive testing for report generation...")
        results = tester.test_comprehensive_recovery()
        
        # Generate comprehensive analysis report
        print("Generating comprehensive analysis report...")
        report = tester.generate_recovery_analysis_report(results)
        
        # Display the report
        print("\n" + "=" * 80)
        print("GENERATED RECOVERY ANALYSIS REPORT")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        # Save report to file
        report_filename = f"usb4_recovery_report_{int(time.time())}.txt"
        try:
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_filename}")
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    finally:
        tester.cleanup()
    
    print("\nRecovery report generation completed!")


def main():
    """Main demonstration function"""
    print("USB4 Link Recovery Testing Demonstration")
    print("This demo showcases comprehensive USB4 link recovery testing capabilities")
    print()
    
    try:
        # Run all demonstrations
        demonstrate_basic_error_recovery()
        demonstrate_comprehensive_recovery_testing()
        demonstrate_recovery_timing_analysis()
        demonstrate_effectiveness_measurement()
        demonstrate_recovery_report_generation()
        
        print("\n" + "=" * 60)
        print("All USB4 Link Recovery Testing Demonstrations Completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
