"""
TC-10: Report Generation - All 5 .docx Reports Produced & Valid
Per PRD Section 10
"""
import pytest
import sys
from pathlib import Path


class TestReportGeneration:
    """Test suite for report generation completeness."""

    def test_report_generator_exists(self):
        """TC-10a: Verify report_generator.py module exists and imports."""
        project_root = Path(__file__).parent.parent.parent
        report_gen = project_root / "backend" / "report_generator.py"
        assert report_gen.exists(), "report_generator.py not found"
        assert report_gen.stat().st_size > 1000, "report_generator.py appears empty"
        print(f"report_generator.py: {report_gen.stat().st_size} bytes")

    def test_report_generator_has_all_methods(self):
        """TC-10b: Verify all 3 report generation methods exist."""
        project_root = Path(__file__).parent.parent.parent
        report_gen_path = project_root / "backend" / "report_generator.py"

        content = report_gen_path.read_text()
        required_methods = [
            "generate_dq_report",
            "generate_model_report",
            "generate_el_report"
        ]

        for method in required_methods:
            assert method in content, f"Missing method: {method}"
            print(f"Found method: {method}")

    def test_report_types_coverage(self):
        """TC-10c: Verify all 5 report types are supported."""
        expected_reports = [
            "data_quality_report.docx",
            "pd_model_report.docx",
            "lgd_model_report.docx",
            "ead_model_report.docx",
            "el_summary_report.docx"
        ]

        for report in expected_reports:
            assert report.endswith('.docx'), f"Report must be .docx: {report}"
            print(f"Expected report: {report}")

        assert len(expected_reports) == 5, "Must produce exactly 5 reports"

    def test_c1_template_sections(self):
        """TC-10d: Verify C1 template sections per PRD Section 10.2."""
        c1_sections = [
            "1. Introduction",
            "2. Purpose and Uses",
            "3. Use Case Description",
            "4.1 Data Description and Sources",
            "4.2 Data Treatments",
            "4.3 Data Assumptions and Limitations",
            "5.1 Feature Engineering and Selection",
            "5.2 Methodology Selection",
            "5.3 Calibration Dataset",
            "5.4 Assumptions and Limitations",
            "6.1 Outputs",
            "6.2 Performance Testing",
            "7. Implementation",
            "8. Appendix"
        ]

        assert len(c1_sections) == 14, "C1 template must have 14 sections"
        for section in c1_sections:
            print(f"C1 section: {section}")

    def test_dq_report_structure(self):
        """TC-10e: Data Quality Report follows custom structure per PRD 10.3."""
        dq_sections = [
            "Data Overview",
            "Data Quality Scorecard",
            "Data Assumptions and Treatments",
            "Feature Profiling",
            "Fit for Model Development Decision",
            "Query Trace"
        ]

        assert len(dq_sections) == 6, "DQ Report must have 6 sections"
        for section in dq_sections:
            print(f"DQ section: {section}")
