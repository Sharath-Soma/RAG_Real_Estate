"""Session export module for exporting conversations to multiple formats."""

from datetime import datetime
from typing import List, Dict, Any, Literal
import json


class SessionExporter:
    """Export conversation sessions to TXT, Markdown, and JSON formats."""

    def __init__(self):
        """Initialize the exporter."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_txt(self, messages: List[Dict[str, Any]], filename: str = None) -> str:
        """Export conversation to plain text format."""
        if filename is None:
            filename = f"conversation_{self.timestamp}.txt"

        lines = [
            "=" * 70,
            f"CONVERSATION EXPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            lines.append(f"{i}. [{role}] {timestamp}")
            lines.append("-" * 70)
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def export_markdown(self, messages: List[Dict[str, Any]], filename: str = None) -> str:
        """Export conversation to Markdown format."""
        if filename is None:
            filename = f"conversation_{self.timestamp}.md"

        lines = [
            "# Conversation Export",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            if role == "user":
                lines.append(f"## 👤 User ({timestamp})")
            elif role == "assistant":
                lines.append(f"## 🤖 Assistant ({timestamp})")
            else:
                lines.append(f"## {role.capitalize()} ({timestamp})")

            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def export_json(self, messages: List[Dict[str, Any]], filename: str = None) -> str:
        """Export conversation to JSON format."""
        if filename is None:
            filename = f"conversation_{self.timestamp}.json"

        export_data = {
            "export_date": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def get_filename(self, format_type: Literal["txt", "md", "json"]) -> str:
        """Generate appropriate filename for format."""
        extensions = {
            "txt": "txt",
            "md": "md",
            "json": "json",
        }
        ext = extensions.get(format_type, "txt")
        return f"conversation_{self.timestamp}.{ext}"

    def export_pdf(self, messages: List[Dict[str, Any]]) -> bytes:
        """Export conversation to a simple professional PDF report."""
        # Note: we use a basic text layout to guarantee wrapping across pages.
        # ReportLab handles pagination by measuring line height and starting new pages.
        buffer = []

        # Lazy import to keep TXT/MD/JSON export working even when reportlab isn't installed.
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except ModuleNotFoundError:
            raise RuntimeError(
                "PDF export requires the 'reportlab' package. Install it using: pip install reportlab"
            )

        # Create a PDF into an in-memory canvas via reportlab canvas.
        from io import BytesIO

        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=letter)
        width, height = letter

        y = height - 1.0 * inch
        left_margin = 1.0 * inch
        right_margin = 0.8 * inch
        usable_width = width - left_margin - right_margin

        def draw_wrapped(text: str, x: float, y_pos: float, max_width: float, font_name: str, font_size: int):
            c.setFont(font_name, font_size)
            words = text.split()
            lines = []
            line = ""
            for w in words:
                candidate = (line + " " + w).strip()
                if c.stringWidth(candidate, font_name, font_size) <= max_width:
                    line = candidate
                else:
                    if line:
                        lines.append(line)
                    line = w
            if line:
                lines.append(line)
            line_height = font_size * 1.2
            for ln in lines:
                if y_pos < 1.0 * inch:
                    c.showPage()
                    y_pos = height - 1.0 * inch
                    c.setFont(font_name, font_size)
                c.drawString(x, y_pos, ln)
                y_pos -= line_height
            return y_pos

        # Title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, y, "Northstar Realty AI Assistant - Session Export")
        y -= 0.4 * inch

        c.setFont("Helvetica", 10)
        c.drawString(left_margin, y, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 0.3 * inch

        c.setFont("Helvetica", 10)
        c.drawString(left_margin, y, "")
        y -= 0.2 * inch

        for idx, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            timestamp = msg.get("timestamp", "")
            content = msg.get("content", "")

            header = f"{idx}. {role.upper()} ({timestamp})"
            c.setFont("Helvetica-Bold", 11)
            if y < 1.0 * inch:
                c.showPage()
                y = height - 1.0 * inch
            c.drawString(left_margin, y, header)
            y -= 0.25 * inch

            if content:
                y = draw_wrapped(
                    content,
                    left_margin,
                    y,
                    usable_width,
                    "Helvetica",
                    10,
                )

            y -= 0.15 * inch

        c.save()
        packet.seek(0)
        return packet.read()

    @staticmethod
    def prepare_download(content: Any, format_type: str) -> Dict[str, Any]:
        """Prepare content for download with proper mimetype."""
        mimetypes = {
            "txt": "text/plain",
            "md": "text/markdown",
            "json": "application/json",
            "pdf": "application/pdf",
        }
        return {
            "content": content,
            "mimetype": mimetypes.get(format_type, "text/plain"),
            "format": format_type,
        }

