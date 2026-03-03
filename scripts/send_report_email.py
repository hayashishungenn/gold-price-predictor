from __future__ import annotations

import argparse
import os
import re
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the rendered daily gold report via email.")
    parser.add_argument("--report-date", type=str, default=date.today().isoformat(), help="Report date in YYYY-MM-DD format.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Project root path.")
    return parser.parse_args()


def _read_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_report_files(project_root: Path, report_date: str) -> tuple[str, str]:
    daily_dir = project_root / "reports" / "daily"
    html_path = daily_dir / f"{report_date}_gold_report.html"
    md_path = daily_dir / f"{report_date}_gold_report.md"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing html report: {html_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"Missing markdown report: {md_path}")
    html_body = html_path.read_text(encoding="utf-8")
    plain_text = md_path.read_text(encoding="utf-8")
    return html_body, plain_text


def _strip_non_email_sections(html_body: str) -> str:
    body = re.sub(r"<section>\s*<h2>图表附录</h2>.*?</section>", "", html_body, flags=re.S)
    body = re.sub(r"<img[^>]*>", "", body, flags=re.I)
    return body


def main() -> None:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    sender = _read_required_env("EMAIL_SENDER")
    password = _read_required_env("EMAIL_PASSWORD")
    receivers = [item.strip() for item in _read_required_env("EMAIL_RECEIVERS").split(",") if item.strip()]

    html_body, plain_text = _load_report_files(project_root=project_root, report_date=args.report_date)
    html_body = _strip_non_email_sections(html_body)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"黄金决策仪表盘 {args.report_date}"
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)
    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())

    print(f"email_sent {','.join(receivers)}")


if __name__ == "__main__":
    main()
