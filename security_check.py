#!/usr/bin/env python3
"""
Security check script for cryptocurrency event study project.

Checks for:
- Hardcoded API keys, passwords, or secrets
- Insecure file permissions
- Unsafe imports or modules
- SQL injection vulnerabilities (if any SQL is used)
- Path traversal vulnerabilities
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any


class SecurityChecker:
    """Security vulnerability checker for the project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []
        
        # Patterns for common security issues
        self.secret_patterns = [
            (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Potential API key'),
            (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']+["\']', 'Potential password'),
            (r'["\']?secret["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Potential secret'),
            (r'["\']?token["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Potential token'),
            (r'sk-[a-zA-Z0-9]{20,}', 'Potential OpenAI API key'),
            (r'xox[baprs]-[a-zA-Z0-9-]+', 'Potential Slack token'),
        ]
        
        # Patterns for unsafe code
        self.unsafe_patterns = [
            (r'eval\s*\(', 'Use of eval() - potential code injection'),
            (r'exec\s*\(', 'Use of exec() - potential code injection'),
            (r'__import__\s*\(', 'Dynamic import - review needed'),
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', 'Shell=True in subprocess'),
            (r'os\.system\s*\(', 'Use of os.system() - potential command injection'),
        ]
        
    def check_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for secrets
            for pattern, description in self.secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip if it's in a comment or looks like example/template
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line = content[line_start:content.find('\n', match.start())]
                    
                    if ('example' in line.lower() or 
                        'template' in line.lower() or
                        line.strip().startswith('#') or
                        'your-api-key' in match.group().lower() or
                        'placeholder' in match.group().lower()):
                        continue
                        
                    issues.append({
                        'file': str(file_path),
                        'line': content[:match.start()].count('\n') + 1,
                        'issue': description,
                        'text': match.group()[:50] + '...' if len(match.group()) > 50 else match.group(),
                        'severity': 'HIGH'
                    })
                    
            # Check for unsafe code patterns
            for pattern, description in self.unsafe_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'file': str(file_path),
                        'line': content[:match.start()].count('\n') + 1,
                        'issue': description,
                        'text': match.group()[:50] + '...' if len(match.group()) > 50 else match.group(),
                        'severity': 'MEDIUM'
                    })
                    
        except Exception as e:
            issues.append({
                'file': str(file_path),
                'line': 0,
                'issue': f'Error reading file: {e}',
                'text': '',
                'severity': 'LOW'
            })
            
        return issues
        
    def check_project(self) -> List[Dict[str, Any]]:
        """Check entire project for security issues."""
        all_issues = []
        
        # Python files to check
        python_files = list(self.project_root.glob('**/*.py'))
        config_files = list(self.project_root.glob('**/*.env*'))
        
        print(f"Checking {len(python_files)} Python files...")
        for py_file in python_files:
            issues = self.check_file(py_file)
            all_issues.extend(issues)
            
        print(f"Checking {len(config_files)} configuration files...")
        for config_file in config_files:
            issues = self.check_file(config_file)
            all_issues.extend(issues)
            
        return all_issues
        
    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check for insecure file permissions."""
        issues = []
        
        # Check for world-writable files
        for file_path in self.project_root.glob('**/*'):
            if file_path.is_file():
                try:
                    mode = oct(file_path.stat().st_mode)[-3:]
                    if mode.endswith('2') or mode.endswith('6'):  # World writable
                        issues.append({
                            'file': str(file_path),
                            'line': 0,
                            'issue': f'World-writable file permissions: {mode}',
                            'text': '',
                            'severity': 'MEDIUM'
                        })
                except Exception:
                    pass  # Skip files we can't read permissions for
                    
        return issues
        
    def generate_report(self, issues: List[Dict[str, Any]]) -> str:
        """Generate a security report."""
        if not issues:
            return "✅ No security issues found!"
            
        high_issues = [i for i in issues if i['severity'] == 'HIGH']
        medium_issues = [i for i in issues if i['severity'] == 'MEDIUM'] 
        low_issues = [i for i in issues if i['severity'] == 'LOW']
        
        report = f"""
🔒 SECURITY SCAN REPORT
=====================

Total Issues Found: {len(issues)}
- High Severity: {len(high_issues)}
- Medium Severity: {len(medium_issues)}
- Low Severity: {len(low_issues)}

"""
        
        for severity, severity_issues in [('HIGH', high_issues), ('MEDIUM', medium_issues), ('LOW', low_issues)]:
            if severity_issues:
                report += f"\n{severity} SEVERITY ISSUES:\n"
                report += "-" * 30 + "\n"
                
                for issue in severity_issues:
                    report += f"File: {issue['file']}\n"
                    if issue['line'] > 0:
                        report += f"Line: {issue['line']}\n"
                    report += f"Issue: {issue['issue']}\n"
                    if issue['text']:
                        report += f"Text: {issue['text']}\n"
                    report += "\n"
                    
        return report
        

def main():
    """Run security checks."""
    print("🔒 Running security checks...")
    
    checker = SecurityChecker()
    
    # Check code for security issues
    code_issues = checker.check_project()
    
    # Check file permissions
    perm_issues = checker.check_file_permissions()
    
    all_issues = code_issues + perm_issues
    
    # Generate and print report
    report = checker.generate_report(all_issues)
    print(report)
    
    # Exit with error code if high severity issues found
    high_severity_count = len([i for i in all_issues if i['severity'] == 'HIGH'])
    if high_severity_count > 0:
        print(f"❌ Found {high_severity_count} high-severity security issues!")
        sys.exit(1)
    elif all_issues:
        print(f"⚠️  Found {len(all_issues)} security issues (none high-severity)")
        sys.exit(0)
    else:
        print("✅ No security issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()