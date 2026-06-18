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
        repository = e.get("issue_repository", {}).get("nameWithOwner", "")
        issue_no = e.get("issue_no", "")
        issues_rows.append({
            "issue_no": issue_no,
            "issue_key": f"{repository}#{issue_no}",
            "issue_url": e.get("issue_url", ""),
            "issue_title": e.get("issue_title", ""),
            "issue_body": e.get("issue_body", ""),
            "repository": repository,
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
    # There are issues which are completely the same
    issues_df = issues_df.drop_duplicates()

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
        repository = e.get("issue_repository", {}).get("nameWithOwner", "")
        issue_no = e.get("issue_no", "")
        issue_key = f"{repository}#{issue_no}"
        for node in e.get("timelineItems", {}).get("nodes", []):
            pr = node.get("pull_info", {})
            if pr:
                links_rows.append({
                    "issue_url": e.get("issue_url", ""),
                    "issue_key": issue_key,
                    "pull_url": pr.get("pull_url"),
                })
    links_df = pd.DataFrame(links_rows)

    return issues_df, prs_df, links_df


def normalise_labels(issues_df, mapping_csv=None):
    """
    Returns a copy of issues_df with the 'labels' column replaced by
    pipe-separated normalised paths from label_mapping.csv
    (e.g. 'area/platform-core/engine|area/ui').
    Labels not present in the mapping are dropped.

    Parameters
    ----------
    issues_df : pd.DataFrame
    mapping_csv : str or None
        Path to label_mapping.csv. Defaults to label_mapping.csv in the
        same directory as this file.
    """
    if mapping_csv is None:
        mapping_csv = os.path.join(os.path.dirname(__file__), "label_mapping.csv")

    mapping = pd.read_csv(mapping_csv)
    label_map = dict(zip(mapping["raw_label"].str.strip(), mapping["normalized_path"].str.strip()))

    def _normalise(raw_labels):
        if not raw_labels or str(raw_labels) == "nan":
            return ""
        normalised = {
            label_map[l.strip()]
            for l in str(raw_labels).split("|")
            if l.strip() in label_map
        }
        return "|".join(sorted(normalised))

    result = issues_df.copy()
    result["labels"] = issues_df["labels"].apply(_normalise)
    return result


def construct_issue_links(issues_df, links_df):
    """
    Returns a DataFrame with columns [issue_key, issue_no, related_issue_keys]
    where related_issue_keys is a list of other repo-qualified issues that share
    at least one PR.
    """
    url_to_key = issues_df.set_index("issue_url")["issue_key"].to_dict()

    links = links_df.copy()
    links["issue_key"] = links["issue_url"].map(url_to_key)
    pr_to_issue_keys = links.groupby("pull_url")["issue_key"].apply(list)

    related_issues = {key: set() for key in issues_df["issue_key"]}
    for issue_key, pull_url in zip(links["issue_key"], links["pull_url"]):
        for related_key in pr_to_issue_keys[pull_url]:
            if related_key != issue_key:
                related_issues[issue_key].add(related_key)

    result = pd.DataFrame([
        {
            "issue_key": issue_key,
            "related_issue_keys": sorted(related),
        }
        for issue_key, related in related_issues.items()
    ])
    return result
