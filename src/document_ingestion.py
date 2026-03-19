import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # pyright: ignore
from PIL import Image # pyright: ignore
from docx import Document # pyright: ignore

logger = logging.getLogger(__name__)

def _import_pdf():
    try:
        return fitz
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

def _import_docx():
    try:
        return Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

def _import_pil():
    try:
        return Image
    except ImportError:
        raise ImportError("Install Pillow: pip install pillow")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:

    if not text.strip():
        return []
    tokens = text.split()
    chunks, i = [], 0
    while i < len(tokens):
        chunk = " ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _pdf_chunks(path: str, source: str) -> List[Dict[str, Any]]:
    fitz = _import_pdf()
    doc  = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc, 1):
        # -- text
        text = page.get_text("text").strip()
        if text:
            for chunk in chunk_text(text):
                chunks.append({
                    "text": chunk,
                    "source": f"{Path(path).name} · p{page_num}",
                    "type": "text",
                    "has_image": False,
                    "image_path": None,
                    "meta": {"file": path, "page": page_num, "source": source},
                })

        # -- embedded images
        for img_idx, img_ref in enumerate(page.get_images(full=True)):
            try:
                xref = img_ref[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                ext = base_img["ext"]
                img_save = Path(path).parent / f"__img_{Path(path).stem}_p{page_num}_{img_idx}.{ext}"
                img_save.write_bytes(img_bytes)

                # caption = surrounding text (heuristic)
                caption = text[:200] if text else f"Image on page {page_num}"
                chunks.append({
                    "text": caption,
                    "source": f"{Path(path).name} · p{page_num} img{img_idx}",
                    "type": "image",
                    "has_image": True,
                    "image_path": str(img_save),
                    "meta": {"file": path, "page": page_num, "source": source},
                })
            except Exception as e:
                logger.warning(f"Could not extract image from {path}: {e}")

    doc.close()
    return chunks


def _image_chunks(path: str, source: str) -> List[Dict[str, Any]]:
    """Treat standalone images as single chunks with empty text."""
    return [{
        "text": f"[Image: {Path(path).name}]",
        "source": Path(path).name,
        "type": "image",
        "has_image": True,
        "image_path": path,
        "meta": {"file": path, "source": source},
    }]


def _docx_chunks(path: str, source: str) -> List[Dict[str, Any]]:
    Document = _import_docx()
    doc = Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    chunks = []
    for chunk in chunk_text(full_text):
        chunks.append({
            "text": chunk,
            "source": Path(path).name,
            "type": "text",
            "has_image": False,
            "image_path": None,
            "meta": {"file": path, "source": source},
        })
    return chunks


SUPPORTED = {
    ".pdf": _pdf_chunks,
    ".png": _image_chunks,
    ".jpg": _image_chunks,
    ".jpeg": _image_chunks,
    ".docx": _docx_chunks,
}


def ingest_documents(directory: str, source: str = "local") -> List[Dict[str, Any]]:
    
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist — generating sample docs.")
        _create_sample_docs(directory)

    all_chunks: List[Dict[str, Any]] = []
    for fpath in sorted(directory.rglob("*")):
        ext = fpath.suffix.lower()
        if ext in SUPPORTED:
            try:
                logger.info(f"Ingesting {fpath}")
                chunks = SUPPORTED[ext](str(fpath), source)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to ingest {fpath}: {e}")

    logger.info(f"Total chunks ingested: {len(all_chunks)}")
    return all_chunks


def _create_sample_docs(directory: Path):
    
    directory.mkdir(parents=True, exist_ok=True)

    sample_texts = {
        "hr_leave_policy.txt": """
        HR Leave Policy — Intranet Document
        =====================================
        Annual Leave: Employees are entitled to 18 days of paid annual leave per year.
        Sick Leave: Up to 12 days of paid sick leave per annum.
        Maternity Leave: 26 weeks as per the Maternity Benefit Act.
        Paternity Leave: 5 days.
        Leave Without Pay (LWP): Requires HR approval in advance.
        Carry Forward: Up to 15 days of annual leave can be carried forward.
        Leave Application: Must be submitted via the HR portal at least 3 days prior.
        Emergency Leave: Up to 3 days without prior approval; documentation required within 7 days.
        """,
        "it_security_policy.txt": """
        IT Security Policy — Internal Use Only
        ========================================
        Password Policy: Minimum 12 characters, alphanumeric + special character.
        Multi-Factor Authentication (MFA) is mandatory for all VPN and cloud access.
        Data Classification: Confidential, Internal, Public — handle accordingly.
        Incident Reporting: All security incidents must be reported to security@company.com within 1 hour.
        Device Management: Only company-approved devices may access internal systems.
        Remote Work: Use company VPN at all times when accessing intranet resources remotely.
        Software Installation: No unauthorized software. Submit requests via IT Help Desk.
        Network Architecture: Segmented VLAN design — production, dev, and admin zones isolated.
        """,
        "product_roadmap_q3.txt": """
        Product Roadmap Q3 2025 — Confidential
        ========================================
        Feature 1: AI-Powered Search (Qwen multimodal integration) — ETA July 2025
        Feature 2: Unified Dashboard for analytics — ETA August 2025
        Feature 3: Mobile App v2.0 with offline support — ETA September 2025
        Feature 4: Automated Report Generation — ETA Q4 carryover
        Infrastructure: Migrate all services to Kubernetes — In Progress
        ML Pipeline: Deploy Qwen-VL model for document understanding — Priority
        Database: Shard PostgreSQL cluster for 10x scale — ETA August 2025
        """,
        "network_architecture.txt": """
        Network Architecture Overview
        ==============================
        The intranet follows a three-tier architecture:
        - Edge Layer: CDN + WAF (Cloudflare) for external traffic
        - Application Layer: Load-balanced Nginx → Kubernetes pods
        - Data Layer: PostgreSQL (primary + replicas), Redis cache, S3 for object storage
        VPN: WireGuard-based VPN for remote access. Tunnel traffic encrypted AES-256.
        Monitoring: Prometheus + Grafana dashboards at internal.monitor.company.net
        DNS: Internal DNS resolves *.intranet.company.com via CoreDNS on Kubernetes.
        Firewall Rules: Whitelist-only for admin ports (22, 5432, 6379).
        Diagram: See attached network_topology_q2.png for visual reference.
        """,
        "onboarding_checklist.txt": """
        New Employee Onboarding Checklist
        ===================================
        Week 1:
        - [ ] Collect ID badge from reception
        - [ ] Setup company laptop (IT will assist)
        - [ ] Configure MFA on company accounts
        - [ ] Complete compliance training on Intranet > Learning Portal
        - [ ] Meet with your team lead and HR buddy
        Week 2:
        - [ ] Shadow senior team members for project context
        - [ ] Access granted to Jira, Confluence, GitHub, and Slack
        - [ ] Submit signed NDA and Code of Conduct to HR
        Month 1:
        - [ ] Complete first sprint with team
        - [ ] 30-day review with manager
        """,
    }

    for filename, content in sample_texts.items():
        (directory / filename).write_text(content.strip(), encoding="utf-8")

    logger.info(f"Sample documents created in {directory}")

