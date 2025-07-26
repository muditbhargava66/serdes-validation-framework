#!/usr/bin/env python3
"""
SerDes Validation Framework Examples Index

This script provides an interactive menu to help users navigate and run
the various examples available in the framework.

Usage:
    python examples/example_index.py
"""

import subprocess
import sys
from pathlib import Path

# Example categories and descriptions
EXAMPLES = {
    "Quick Start": [
        ("usb4_quick_start.py", "Quick USB4 validation introduction", "Beginner"),
        ("data_analysis_example.py", "Basic data analysis concepts", "Beginner"),
    ],
    
    "Protocol-Specific": [
        ("pcie_example.py", "Complete PCIe 6.0 validation workflow", "Intermediate"),
        ("eth_224g_example.py", "224G Ethernet validation", "Intermediate"),
        ("usb4_basic_validation_example.py", "Comprehensive USB4 validation", "Intermediate"),
        ("pam4_analysis_example.py", "PAM4 signal analysis techniques", "Intermediate"),
    ],
    
    "USB4/Thunderbolt 4": [
        ("usb4_thunderbolt_certification_example.py", "Thunderbolt 4 certification testing", "Advanced"),
        ("usb4_tunneling_example.py", "Multi-protocol tunneling validation", "Advanced"),
        ("usb4_jitter_analysis_demo.py", "Advanced jitter analysis with SSC", "Advanced"),
        ("usb4_power_management_demo.py", "Power state management validation", "Advanced"),
        ("usb4_link_recovery_demo.py", "Link recovery and error handling", "Advanced"),
    ],
    
    "Advanced Framework": [
        ("multi_protocol_comparison.py", "Cross-protocol performance comparison", "Advanced"),
        ("framework_integration_example.py", "Unified validation framework usage", "Advanced"),
        ("test_sequence_example.py", "Advanced test sequence automation", "Advanced"),
    ]
}

def print_header():
    """Print the header"""
    print("=" * 80)
    print("SerDes Validation Framework v1.4.0 - Examples Index")
    print("=" * 80)
    print()

def print_examples_menu():
    """Print the examples menu"""
    print("Available Examples:")
    print()
    
    example_num = 1
    example_map = {}
    
    for category, examples in EXAMPLES.items():
        print(f"📁 {category}")
        print("-" * (len(category) + 2))
        
        for filename, description, level in examples:
            level_icon = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}[level]
            print(f"  {example_num:2d}. {level_icon} {filename}")
            print(f"      {description}")
            print(f"      Level: {level}")
            print()
            
            example_map[example_num] = filename
            example_num += 1
    
    return example_map

def print_usage_info():
    """Print usage information"""
    print("Usage Information:")
    print("-" * 17)
    print("🟢 Beginner:    Start here for basic concepts")
    print("🟡 Intermediate: Requires understanding of SerDes concepts")
    print("🔴 Advanced:     Complex examples for experienced users")
    print()
    print("Most examples support --help for detailed options")
    print("Use --mock flag to run without hardware")
    print()

def run_example(filename: str):
    """Run a selected example"""
    example_path = Path(__file__).parent / filename
    
    if not example_path.exists():
        print(f"❌ Error: Example file '{filename}' not found!")
        return False
    
    print(f"🚀 Running: {filename}")
    print("-" * (len(filename) + 10))
    print()
    
    try:
        # Run the example
        result = subprocess.run([sys.executable, str(example_path)], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Example '{filename}' completed successfully!")
        else:
            print(f"\n❌ Example '{filename}' failed with exit code {result.returncode}")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Example '{filename}' interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        return False

def show_example_help(filename: str):
    """Show help for a specific example"""
    example_path = Path(__file__).parent / filename
    
    if not example_path.exists():
        print(f"❌ Error: Example file '{filename}' not found!")
        return
    
    print(f"📖 Help for: {filename}")
    print("-" * (len(filename) + 12))
    print()
    
    try:
        result = subprocess.run([sys.executable, str(example_path), "--help"], 
                              capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        else:
            print("No help information available for this example.")
            
    except Exception as e:
        print(f"❌ Error getting help: {e}")

def main():
    """Main interactive menu"""
    print_header()
    
    while True:
        example_map = print_examples_menu()
        print_usage_info()
        
        print("Options:")
        print("  Enter number (1-{}) to run an example".format(len(example_map)))
        print("  Enter 'h<number>' to show help for an example (e.g., 'h1')")
        print("  Enter 'q' to quit")
        print()
        
        try:
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            
            elif choice.startswith('h') and len(choice) > 1:
                # Show help
                try:
                    num = int(choice[1:])
                    if num in example_map:
                        show_example_help(example_map[num])
                    else:
                        print(f"❌ Invalid example number: {num}")
                except ValueError:
                    print("❌ Invalid help command. Use format 'h<number>' (e.g., 'h1')")
            
            elif choice.isdigit():
                # Run example
                num = int(choice)
                if num in example_map:
                    filename = example_map[num]
                    print()
                    run_example(filename)
                else:
                    print(f"❌ Invalid choice: {num}")
            
            else:
                print("❌ Invalid input. Please enter a number, 'h<number>', or 'q'")
            
            print()
            input("Press Enter to continue...")
            print("\n" * 2)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
