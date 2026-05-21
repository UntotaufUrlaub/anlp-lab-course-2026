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


_LABEL_MAP = {
    "bug":            ["bug", "type: bug", "[Type] Bug", "type:bug", "t/bug :bug:", "t/bug",
                       "🐞 Bug", "🐛 Bug", ":bug: bug", "kind/bug", "Issue-Bug", "T-bug",
                       "bug report", "Defect", "T-Defect", "Type: Confirmed bug",
                       "report/crash report", "type: bug/fix", "type: accepted/bug",
                       "[Type] Broken Window"],
    "crash":          ["crash", "[Type] Crash", "Type: Crash", "severe: crash"],
    "regression":     ["regression", "severe: regression"],
    "enhancement":    ["enhancement", "feature", "feature request", "[Type] Enhancement",
                       "type: enhancement", "t/enhancement ➕", "Type: Improvement", "improvement",
                       "Type: Feature Request", "Type: Feature", "severe: new feature", "proposal",
                       "T-Enhancement", "New feature", "feature-request", "Performance",
                       "New architecture"],
    "task":           ["task", "type: task", "[Type] Task", "refactor", "testing"],
    "question":       ["question", "discussion", "support"],
    "documentation":  ["documentation", "docs", "comp: docs", "d: api docs"],
    "priority:critical": ["P0", "blocker", "release blocker", "HIGH PRIORITY", "[Pri] Blocking",
                          "e2e test blocker", "critical"],
    "priority:high":     ["P1", "priority: high", "high-priority", "[Pri] High",
                          "Priority: Essential", "priority-P1", "severity: high", "high-severity",
                          "priority: important", "high", "p1", "Priority-High",
                          "freq3: high", "priority: p1"],
    "priority:medium":   ["P2", "priority: medium", "[Pri] Medium", "medium-priority",
                          "medium-severity", "priority: p2", "p2-high", "freq2: medium",
                          "Priority-Medium", "priority/P2"],
    "priority:low":      ["P3", "P4", "priority: p3", "[Pri] Low", "Priority: Nice-to-have",
                          "p3", "freq1: low", "Priority: Low", "low-severity"],
    "status:in_progress": ["in progress", "in-progress", "assigned", "triaged", "approved",
                           "confirmed", "accepted", "reproduced", "Status: Not started",
                           "Needs Triage", "needs info"],
    "status:qa":          ["[QA]:Verified fixed", "[QA]:Normal issue", "[QA]:Major issue",
                           "[QA]:Minor issue", "[QA]:Blocker issue", "[QA]:Caught_by_exploratory",
                           "Q-verified", "eng:qa:verified", "state: verified fixed",
                           "QA Pass-Win64", "QA Pass-Linux", "QA Pass-macOS",
                           "verified", "QA-verified", "QA/Yes"],
    "status:resolved":    ["fixed", "resolution: fixed", "released", "Done", "PR exists",
                           "has pr", "pr-merged", "Patch available",
                           "waiting for PR to land (fixed)", "Resolution: PR Submitted",
                           "has-pr", "state: has PR"],
    "status:stale":       ["Stale", "outdated", "frozen-due-to-age", "Resolution: Locked",
                           "Duplicate", "backlog", "wontfix"],
    "platform:android":   ["Android", "platform: android", "p/android", "platform-android",
                           "platform:android", "OS: Android", "P-android", "[OS] Android",
                           "platform/android", "mobile-app"],
    "platform:ios":       ["iOS", "platform: ios", "platform-ios", "os: iOS", "p/iOS 🍎"],
    "platform:web":       ["platform: web", "platform-web", "web", "Mobile Web", "platform:web"],
    "platform:linux":     ["Platform: Linux", "platform-linux", "os: Linux",
                           "platform: Linux 🐧"],
    "layer:frontend":     ["UI", "ux", "Design", "GUI", "frontend", "front-end", "front end",
                           "CSS", "layout", "Rendering", "ui/ux", "Area-UIUX", "component: ux",
                           "f: material design", "Type: Frontend", "ui-mobile", "[Type] UI Bug",
                           "NUX", "design issue", "ui: CSS", "ux-improvement", "UI-XML/Widgets",
                           "Frontend Design", "Type: UX", "component: legacy frontend",
                           "A-frontend", "subj: ui/ux", "Rendering bug", "a: text input",
                           "accessibility", "a: accessibility"],
    "layer:backend":      ["backend", "server", "API", "comp: server", "Type: Backend",
                           "AREA: server", "crate:server", "server side", "Server issue",
                           "rest-api", "comp: http", "websockets", "Service: Messaging",
                           "Service: Database", "Service: Authentication", "comp: service-worker",
                           "severe: API break", "type: api", "Core REST API Task"],
    "layer:infrastructure": ["Build", "infra", "Infrastructure", "A-build",
                             "Area: App+Library Build", "comp: build & ci", "travis-build",
                             "build-ci", "build system", "area: build", "T-infra"],
    "community:good_first_issue": ["good first issue", "up-for-grabs", "Easy",
                                   "welcome contribute", "Good First Issue!", "E-easy"],
    "community:help_wanted": ["help wanted"],
    "community:bounty":      ["Bounty", "bounty-xs", "bounty-S", "Hacktoberfest", "BOSS",
                              "community-sprint", "PSoC - 2020"],
}

_REVERSE_LABEL_MAP = {raw: canonical for canonical, raws in _LABEL_MAP.items() for raw in raws}


def normalise_labels(issues_df):
    """
    Returns a copy of issues_df with the 'labels' column replaced by
    pipe-separated canonical group names (e.g. 'bug|priority:high|layer:frontend').
    Labels not in the map are dropped.
    """
    def _normalise(raw_labels):
        if not raw_labels or str(raw_labels) == "nan":
            return ""
        canonical = {
            _REVERSE_LABEL_MAP[l.strip()]
            for l in str(raw_labels).split("|")
            if l.strip() in _REVERSE_LABEL_MAP
        }
        return "|".join(sorted(canonical))

    result = issues_df.copy()
    result["labels"] = issues_df["labels"].apply(_normalise)
    return result


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
