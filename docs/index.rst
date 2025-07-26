SerDes Validation Framework Documentation
========================================

Welcome to the comprehensive documentation for the SerDes Validation Framework - a powerful, extensible platform for high-speed serial interface validation and testing.

🚀 What's New
-------------

Latest Features (v1.4.0+)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Complete USB4/Thunderbolt 4 Support** - Full implementation with dual-lane analysis, tunneling validation, and certification testing
* **Advanced Mock Testing Infrastructure** - Comprehensive mock system enabling CI/CD testing without hardware
* **Multi-Protocol Validation** - Unified framework supporting USB4, PCIe, Thunderbolt, and 224G Ethernet
* **Real-time Visualization** - Advanced plotting and monitoring capabilities
* **Automated Reporting** - Certification-ready reports with trend analysis

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   INSTALL
   USAGE
   tutorials/getting_started
   guides/installation
   reference/configuration

.. toctree::
   :maxdepth: 2
   :caption: USB4/Thunderbolt 4
   :hidden:

   usb4/index
   usb4/quickstart
   usb4/api-reference
   usb4/certification/index
   usb4/certification/thunderbolt4
   usb4/certification/intel-requirements
   usb4/examples/basic-validation
   usb4/guides/best-practices

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :hidden:

   api/index
   api/reporting
   api/instrument_control
   api/mock_controller
   api/eth_224g
   api/pam4_analysis
   api/pcie
   api/usb4_thunderbolt

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/mock_testing
   tutorials/224g_validation
   tutorials/pam4_analysis
   tutorials/pcie_validation

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/mock_examples
   examples/reporting_examples
   examples/data_analysis_examples
   examples/mock_controller_examples
   examples/pcie_examples
   examples/testing_examples

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   guides/testing
   guides/cicd
   guides/troubleshooting
   guides/instrument_setup
   guides/environment_vars

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/configuration
   toc

.. toctree::
   :maxdepth: 1
   :caption: Project Info
   :hidden:

   CONTRIBUTING
   README
   DOCUMENTATION_UPDATES

🎯 Key Capabilities
-------------------

USB4/Thunderbolt 4 Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Dual-lane signal analysis with skew measurement
* Complete tunneling protocol validation (PCIe, DisplayPort, USB 3.2)
* Thunderbolt 4 security and daisy-chain testing
* Intel certification compliance testing

Signal Integrity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

* Advanced eye diagram analysis with compliance masks
* Comprehensive jitter decomposition (RJ, DJ, PJ)
* Multi-lane skew analysis and compensation
* Real-time signal quality monitoring

Test Automation
~~~~~~~~~~~~~~~

* Automated test sequence orchestration
* Mock testing for CI/CD environments
* Performance regression testing
* Comprehensive reporting and visualization

🛠️ Framework Architecture
-------------------------

The framework is built with modularity and extensibility in mind::

    SerDes Validation Framework
    ├── Protocol Modules (USB4, PCIe, Ethernet)
    ├── Signal Analysis Engine
    ├── Instrument Control Layer
    ├── Test Automation Framework
    ├── Visualization & Reporting
    └── Mock Testing Infrastructure

📊 Supported Protocols
----------------------

.. list-table::
   :header-rows: 1

   * - Protocol
     - Status
     - Features
   * - USB4 2.0
     - ✅ Complete
     - Dual-lane, tunneling, power management
   * - Thunderbolt 4
     - ✅ Complete
     - Security, daisy-chain, certification
   * - PCIe 6.0
     - ✅ Complete
     - NRZ/PAM4, link training, compliance
   * - 224G Ethernet
     - ✅ Complete
     - PAM4 analysis, equalization
   * - Multi-Protocol
     - ✅ Complete
     - Cross-protocol validation

🧪 Testing Infrastructure
-------------------------

The framework includes a comprehensive testing infrastructure:

* **62+ Passing Tests** - Extensive test coverage
* **Mock Testing Support** - Test without hardware
* **CI/CD Ready** - Automated testing pipeline
* **Performance Regression** - Trend analysis and monitoring

📈 Getting Started
------------------

1. **Install the Framework**

   .. code-block:: bash

      pip install serdes-validation-framework

2. **Run Your First Test**

   .. code-block:: python

      from serdes_validation_framework import USB4Validator
      
      validator = USB4Validator()
      results = validator.validate_compliance(signal_data)

3. **Generate Reports**

   .. code-block:: python

      from serdes_validation_framework.reporting import USB4TestReporter
      
      reporter = USB4TestReporter()
      report_path = reporter.generate_compliance_report(session_id)

Quick Start Links
~~~~~~~~~~~~~~~~~

New to USB4 validation? Start here:

* :doc:`INSTALL` - Quick installation guide
* :doc:`usb4/quickstart` - USB4 validation walkthrough
* :doc:`usb4/api-reference` - Complete API documentation
* :doc:`tutorials/getting_started` - Your first validation test

🤝 Community & Support
----------------------

* **GitHub Repository**: `SerDes Validation Framework <https://github.com/muditbhargava66/serdes-validation-framework>`_
* **Issue Tracker**: Report bugs and request features
* **Discussions**: Community support and questions
* **Contributing**: See our :doc:`CONTRIBUTING` guide

📄 License
----------

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to start validating?** Check out our :doc:`tutorials/getting_started` to begin your journey with the SerDes Validation Framework.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`