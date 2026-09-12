"""
Resume Extractor
-----------------
A rule-based script that uses regular expressions to extract phone numbers
and email addresses from resume text, then saves the structured results
to output.json.

Usage:
    python resume_extractor.py

This will run the extractor on the sample resumes defined in
`sample_resumes.py` (or inline below) and write `output.json`.

You can also import and use `extract_contact_info()` on any text.
"""

import re
import json
from pathlib import Path


# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------

# Email pattern: standard RFC-ish pattern that covers the vast majority of
# real-world email addresses (local-part@domain.tld)
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

# Phone pattern: handles a wide range of common resume phone formats, e.g.
#   +1 (555) 123-4567
#   555-123-4567
#   555.123.4567
#   5551234567
#   +91 98765 43210
#   (022) 2345 6789
PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)                                  # not preceded by a digit
    (?:\+?\d{1,3}[\s.-]?)?                   # optional country code
    (?:\(\d{2,4}\)[\s.-]?)?                  # optional area code in parens
    (?:\d{2,4}[\s.-])?                       # optional area code / prefix
    \d{3,5}[\s.-]?\d{3,5}                    # main number block (handles 3-5 digit groupings)
    (?!\d)                                   # not followed by a digit
    """,
    re.VERBOSE,
)


def _clean_phone(raw: str) -> str:
    """Normalize whitespace in a matched phone string."""
    return re.sub(r"\s+", " ", raw.strip())


def _is_valid_phone(candidate: str) -> bool:
    """
    Filter out false positives (e.g. dates, zip codes, stray numbers)
    by checking the digit count falls in a plausible phone-number range.
    """
    digits = re.sub(r"\D", "", candidate)
    return 7 <= len(digits) <= 15


def extract_emails(text: str) -> list:
    """Return a de-duplicated list of email addresses found in text."""
    found = EMAIL_PATTERN.findall(text)
    # de-duplicate while preserving order
    seen = set()
    emails = []
    for e in found:
        e_clean = e.strip().rstrip(".,;:")
        if e_clean.lower() not in seen:
            seen.add(e_clean.lower())
            emails.append(e_clean)
    return emails


def extract_phones(text: str) -> list:
    """Return a de-duplicated list of valid-looking phone numbers found in text."""
    candidates = PHONE_PATTERN.findall(text)
    seen = set()
    phones = []
    for c in candidates:
        c_clean = _clean_phone(c)
        if not c_clean or not _is_valid_phone(c_clean):
            continue
        key = re.sub(r"\D", "", c_clean)
        if key not in seen:
            seen.add(key)
            phones.append(c_clean)
    return phones


def extract_contact_info(text: str) -> dict:
    """Extract emails and phone numbers from a single resume text."""
    return {
        "emails": extract_emails(text),
        "phone_numbers": extract_phones(text),
    }


# --------------------------------------------------------------------------
# Sample resumes for testing (>= 3 required by the task)
# --------------------------------------------------------------------------

SAMPLE_RESUMES = {
    "resume_1.txt": """
        John Doe
        Software Engineer

        Contact: john.doe1990@gmail.com | Phone: +1 (555) 123-4567
        LinkedIn: linkedin.com/in/johndoe

        Summary
        -------
        Experienced backend engineer with 6 years building scalable APIs.

        Experience
        ----------
        Senior Software Engineer, Acme Corp (2021 - Present)
        - Led a team of 4 engineers.
        - Reach me anytime at john.doe.work@acme.com or call 555-987-6543.
    """,
    "resume_2.txt": """
        Priya Sharma
        Data Scientist

        Email: priya.sharma@outlook.com
        Mobile: +91 98765 43210
        Alternate contact: (022) 2345 6789

        Education
        ---------
        M.Sc. Statistics, University of Delhi, 2019

        Skills: Python, SQL, Machine Learning, NLP
    """,
    "resume_3.txt": """
        Michael O'Brien
        Product Manager

        michael.obrien+jobs@company.co.uk
        Tel: 0044 20 7946 0958
        Secondary phone: 07911.123456

        Profile
        -------
        Product manager with a passion for user-centric design.
        For urgent matters, text 555.222.3333 or email m.obrien@yahoo.com.
    """,
}


def run_on_samples(output_path: str = "output.json") -> dict:
    """Run the extractor on all sample resumes and write output.json."""
    results = {}
    for name, text in SAMPLE_RESUMES.items():
        results[name] = extract_contact_info(text)

    out_file = Path(output_path)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Extraction complete. Results saved to: {out_file.resolve()}")
    return results


if __name__ == "__main__":
    data = run_on_samples()
    print(json.dumps(data, indent=2))
