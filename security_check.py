#!/usr/bin/env python
"""
Security check script for the cryptocurrency event study repository.
Checks for common security issues before making the repository public.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class SecurityChecker:
    """Check for security issues in the repository."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues_found = []
        self.warnings = []
        
    def check_env_file(self) -> bool:
        """Check if .env file exists (it shouldn't in a public repo)."""
        env_file = self.root_dir / '.env'
        if env_file.exists():
            content = env_file.read_text()
            if any(key in content for key in ['API_KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                self.issues_found.append(
                    "CRITICAL: .env file contains sensitive information and should not be committed!"
                )
                return False
        return True
    
    def check_gitignore(self) -> bool:
        """Check if .gitignore properly excludes sensitive files."""
        gitignore = self.root_dir / '.gitignore'
        if not gitignore.exists():
            self.issues_found.append("No .gitignore file found!")
            return False
            
        content = gitignore.read_text()
        required_patterns = ['.env', '*.key', '*.pem', '*.log']
        missing = []
        
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)
                
        if missing:
            self.warnings.append(
                f"Consider adding these patterns to .gitignore: {', '.join(missing)}"
            )
        return True
    
    def scan_for_secrets(self) -> List[Tuple[str, int, str]]:
        """Scan files for potential secrets or API keys."""
        secret_patterns = [
            (r'[A-Za-z0-9]{32,}', 'Potential API key or token'),
            (r'(?i)(api[_\s]?key|secret|token|password)\s*=\s*["\'][^"\']+["\']', 
             'Hardcoded credential'),
            (r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+', 'Bearer token'),
            (r'[A-Za-z0-9+/]{40,}={0,2}', 'Base64 encoded secret'),
            (r'-----BEGIN (RSA )?PRIVATE KEY-----', 'Private key'),
        ]
        
        findings = []
        extensions_to_check = ['.py', '.json', '.yaml', '.yml', '.txt', '.md']
        exclude_dirs = ['.git', '__pycache__', 'venv', '.venv', 'node_modules']
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions_to_check:
                # Skip if in excluded directory
                if any(exc in str(file_path) for exc in exclude_dirs):
                    continue
                    
                # Skip .env.example (it's supposed to have placeholder keys)
                if file_path.name == '.env.example':
                    continue
                    
                try:
                    content = file_path.read_text(errors='ignore')
                    for line_num, line in enumerate(content.split('\n'), 1):
                        for pattern, description in secret_patterns:
                            if re.search(pattern, line):
                                # Filter out false positives
                                if self._is_false_positive(line, pattern):
                                    continue
                                findings.append((str(file_path), line_num, description))
                except Exception:
                    continue
                    
        return findings
    
    def _is_false_positive(self, line: str, pattern: str) -> bool:
        """Check if a potential secret is actually a false positive."""
        false_positive_indicators = [
            'example', 'placeholder', 'your_', 'xxx', 'todo', 
            'test', 'demo', 'sample', '<', '>'
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in false_positive_indicators)
    
    def check_dependencies(self) -> bool:
        """Check for known vulnerable dependencies."""
        requirements_file = self.root_dir / 'requirements.txt'
        if not requirements_file.exists():
            self.warnings.append("No requirements.txt found")
            return True
            
        # This is a simplified check - in production, use tools like safety or pip-audit
        content = requirements_file.read_text()
        
        # Check for unpinned dependencies
        unpinned = []
        for line in content.split('\n'):
            if line.strip() and not line.startswith('#'):
                if '>=' in line and not ',' in line and not '<' in line:
                    unpinned.append(line.split('>=')[0].strip())
                    
        if unpinned:
            self.warnings.append(
                f"Consider pinning exact versions for: {', '.join(unpinned[:5])}"
            )
        return True
    
    def check_hardcoded_paths(self) -> List[Tuple[str, int]]:
        """Check for hardcoded absolute paths."""
        findings = []
        path_patterns = [
            r'["\']/(home|Users|var|etc|tmp)/[^"\']+["\']',
            r'["\']C:\\\\[^"\']+["\']',
            r'["\']~/[^"\']+["\']',
        ]
        
        for py_file in self.root_dir.rglob('*.py'):
            if '.git' in str(py_file):
                continue
                
            try:
                content = py_file.read_text()
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in path_patterns:
                        if re.search(pattern, line):
                            findings.append((str(py_file), line_num))
            except Exception:
                continue
                
        return findings
    
    def run_all_checks(self) -> bool:
        """Run all security checks."""
        print(f"{BLUE}Running security checks...{RESET}\n")
        
        # Check .env file
        print("Checking for .env file...")
        if not self.check_env_file():
            print(f"{RED}✗ .env file check failed{RESET}")
        else:
            print(f"{GREEN}✓ No sensitive .env file found{RESET}")
        
        # Check .gitignore
        print("\nChecking .gitignore...")
        if self.check_gitignore():
            print(f"{GREEN}✓ .gitignore exists{RESET}")
        
        # Scan for secrets
        print("\nScanning for potential secrets...")
        secrets = self.scan_for_secrets()
        if secrets:
            print(f"{RED}✗ Found {len(secrets)} potential secret(s):{RESET}")
            for file_path, line_num, description in secrets[:10]:  # Show first 10
                print(f"  {file_path}:{line_num} - {description}")
        else:
            print(f"{GREEN}✓ No obvious secrets found{RESET}")
        
        # Check dependencies
        print("\nChecking dependencies...")
        self.check_dependencies()
        print(f"{GREEN}✓ Dependencies checked{RESET}")
        
        # Check hardcoded paths
        print("\nChecking for hardcoded paths...")
        paths = self.check_hardcoded_paths()
        if paths:
            print(f"{YELLOW}⚠ Found {len(paths)} hardcoded path(s):{RESET}")
            for file_path, line_num in paths[:5]:  # Show first 5
                print(f"  {file_path}:{line_num}")
        else:
            print(f"{GREEN}✓ No hardcoded paths found{RESET}")
        
        # Summary
        print(f"\n{BLUE}={'='*50}{RESET}")
        print(f"{BLUE}SECURITY CHECK SUMMARY{RESET}")
        print(f"{BLUE}={'='*50}{RESET}\n")
        
        if self.issues_found:
            print(f"{RED}Critical Issues ({len(self.issues_found)}):{RESET}")
            for issue in self.issues_found:
                print(f"  • {issue}")
            print()
        
        if self.warnings:
            print(f"{YELLOW}Warnings ({len(self.warnings)}):{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if not self.issues_found and not self.warnings:
            print(f"{GREEN}✓ All security checks passed!{RESET}")
            print("Your repository appears ready for public release.")
            return True
        elif not self.issues_found:
            print(f"{YELLOW}⚠ Some warnings found, but no critical issues.{RESET}")
            print("Consider addressing the warnings before making public.")
            return True
        else:
            print(f"{RED}✗ Critical issues found!{RESET}")
            print("Please fix these issues before making the repository public.")
            return False


def main():
    """Main entry point."""
    checker = SecurityChecker()
    success = checker.run_all_checks()
    
    # Additional recommendations
    print(f"\n{BLUE}Additional Recommendations:{RESET}")
    print("1. Regenerate any exposed API keys immediately")
    print("2. Review commit history for sensitive data (use git-filter-branch if needed)")
    print("3. Enable GitHub secret scanning")
    print("4. Set up branch protection rules")
    print("5. Configure CODEOWNERS file for sensitive areas")
    print("6. Consider using pre-commit hooks for ongoing security")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
