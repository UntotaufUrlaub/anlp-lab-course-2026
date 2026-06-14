"""
Validation script to check if the benchmark pipeline is ready to run.
Verifies data files, dependencies, and provides diagnostic information.
"""

import json
from pathlib import Path
import sys

def check_data_files():
    """Check if required data files exist and are valid."""
    print("\n" + "="*60)
    print("CHECKING DATA FILES")
    print("="*60)
    
    docs_path = Path("output/documents.jsonl")
    qrels_path = Path("output/qrels.jsonl")
    
    issues = []
    
    # Check documents.jsonl
    if not docs_path.exists():
        print(f"❌ documents.jsonl not found at {docs_path}")
        issues.append("Missing documents.jsonl")
    else:
        try:
            doc_count = 0
            with open(docs_path, 'r', encoding='utf8') as f:
                for line in f:
                    if line.strip():
                        doc_count += 1
            
            print(f"✅ documents.jsonl valid")
            print(f"   - {doc_count} documents")
        except Exception as e:
            print(f"❌ Error reading documents.jsonl: {e}")
            issues.append(f"documents.jsonl read error: {e}")
    
    # Check qrels.jsonl
    if not qrels_path.exists():
        print(f"❌ qrels.jsonl not found at {qrels_path}")
        issues.append("Missing qrels.jsonl")
    else:
        try:
            qrel_count = 0
            query_count = 0
            queries = set()
            with open(qrels_path, 'r') as f:
                for line in f:
                    if line.strip():
                        qrel = json.loads(line)
                        qrel_count += 1
                        query_id = qrel.get("query_id")
                        if query_id:
                            queries.add(query_id)
            
            query_count = len(queries)
            print(f"✅ qrels.jsonl valid")
            print(f"   - {qrel_count} relevance judgments")
            print(f"   - {query_count} unique queries")
        except Exception as e:
            print(f"❌ Error reading qrels.jsonl: {e}")
            issues.append(f"qrels.jsonl read error: {e}")
    
    return issues


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    dependencies = {
        'rank_bm25': 'rank-bm25',
        'sentence_transformers': 'sentence-transformers',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn',
    }
    
    issues = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (install with: pip install {package_name})")
            issues.append(f"Missing {package_name}")
    
    return issues


def check_script():
    """Check if benchmark script exists."""
    print("\n" + "="*60)
    print("CHECKING BENCHMARK SCRIPT")
    print("="*60)
    
    script_path = Path("code/benchmark.py")
    issues = []
    
    if script_path.exists():
        print(f"✅ benchmark.py found")
    else:
        print(f"❌ benchmark.py not found at {script_path}")
        issues.append("Missing benchmark.py")
    
    return issues


def check_output_directory():
    """Check if output directory exists."""
    print("\n" + "="*60)
    print("CHECKING OUTPUT DIRECTORY")
    print("="*60)
    
    output_dir = Path("results")
    issues = []
    
    if output_dir.exists():
        print(f"✅ Output directory exists: {output_dir}")
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Output directory created: {output_dir}")
        except Exception as e:
            print(f"❌ Could not create output directory: {e}")
            issues.append(f"Cannot create output directory: {e}")
    
    return issues


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " BENCHMARK VALIDATION SCRIPT ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_data_files())
    all_issues.extend(check_dependencies())
    all_issues.extend(check_script())
    all_issues.extend(check_output_directory())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not all_issues:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYou can now run the benchmark:")
        print("  python project_code/benchmark.py")
        return 0
    else:
        print(f"\n❌ FOUND {len(all_issues)} ISSUE(S):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\nPlease resolve the issues above before running the benchmark.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
