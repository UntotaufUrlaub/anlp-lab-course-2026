import pandas as pd
import os
import json


def load_json_dataframes(path):
    """
    Returns (issues_df, prs_df, links_df) extracted from the first JSON file in path.
    Returns (None, None, None) if no JSON file is found.
    """
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not json_files:
        print("No JSON files found in dataset.")
        return None, None, None

    with open(os.path.join(path, json_files[0]), "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total JSON entries: {len(data)}")

    issues_rows = []
    for e in data:
        issues_rows.append({
            "issue_no": e.get("issue_no", ""),
            "issue_url": e.get("issue_url", ""),
            "issue_title": e.get("issue_title", ""),
            "issue_body": e.get("issue_body", ""),
            "repository": e.get("issue_repository", {}).get("nameWithOwner", ""),
            "created_at": e.get("created_at", ""),
            "closed_at": e.get("closed_at", ""),
            "labels": "|".join(
                n["node"]["label"]
                for n in e.get("issue_labels", {}).get("edges", [])
            ),
            "comments": " | ".join(
                n["node"]["comment_body"]
                for n in e.get("issue_comments", {}).get("edges", [])
            ),
        })
    issues_df = pd.DataFrame(issues_rows)

    prs_seen = {}
    for e in data:
        for node in e.get("timelineItems", {}).get("nodes", []):
            pr = node.get("pull_info", {})
            if pr and pr.get("pull_url") not in prs_seen:
                prs_seen[pr["pull_url"]] = {
                    "pull_no": pr.get("pull_no"),
                    "pull_url": pr.get("pull_url"),
                    "pull_title": pr.get("pull_title"),
                    "pull_body": pr.get("pull_body", ""),
                    "repository": pr.get("repository", {}).get("nameWithOwner", ""),
                    "pull_created_at": pr.get("createdAt"),
                    "pull_closed_at": pr.get("closedAt"),
                }
    prs_df = pd.DataFrame(list(prs_seen.values()))

    links_rows = []
    for e in data:
        for node in e.get("timelineItems", {}).get("nodes", []):
            pr = node.get("pull_info", {})
            if pr:
                links_rows.append({
                    "issue_url": e.get("issue_url", ""),
                    "pull_url": pr.get("pull_url"),
                })
    links_df = pd.DataFrame(links_rows)

    return issues_df, prs_df, links_df


def construct_issue_links(issues_df, links_df):
    """
    Returns a DataFrame with columns [issue_no, related_issue_nos] where
    related_issue_nos is a list of other issues that share at least one PR.
    """
    url_to_no = issues_df.set_index("issue_url")["issue_no"].to_dict()

    links = links_df.copy()
    links["issue_no"] = links["issue_url"].map(url_to_no)
    pr_to_issue_nos = links.groupby("pull_url")["issue_no"].apply(list)

    related_issues = {nr: set() for nr in issues_df["issue_no"]}
    for issue_no, pull_url in zip(links["issue_no"], links["pull_url"]):
        for related_no in pr_to_issue_nos[pull_url]:
            if related_no != issue_no:
                related_issues[issue_no].add(related_no)

    result = pd.DataFrame([
        {"issue_no": issue_no, "related_issue_nos": sorted(related)}
        for issue_no, related in related_issues.items()
    ])
    return result
