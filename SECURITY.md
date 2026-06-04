# Security Policy

## Supported versions

earlyon is pre-1.0; security fixes land on the latest released minor version.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than opening a public
issue.

- Email: **sohams.web@gmail.com** with the subject line `earlyon security`.
- Alternatively, use GitHub's
  [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository.

Include, where possible:

- a description of the issue and its impact,
- the version / commit affected,
- steps to reproduce or a proof of concept,
- any suggested remediation.

## What to expect

- Acknowledgement within **5 business days**.
- A triage assessment and severity rating shortly after.
- Coordinated disclosure once a fix is available; credit is offered unless you
  prefer to remain anonymous.

## Scope notes

earlyon loads model checkpoints with `torch.load(..., weights_only=True)` to
prevent arbitrary code execution from untrusted `.pth` files. If you find a way
to bypass that protection, or any other code-execution, data-exfiltration, or
denial-of-service vector in the library, it is in scope.
