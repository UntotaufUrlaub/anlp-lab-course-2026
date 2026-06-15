import json
from collections import Counter, defaultdict


ALLOWED_FIELDS = {
    "Computer Science", "Mathematics", "Engineering", "Medicine", "Biology",
    "Physics", "Economics", "Psychology", "Social Sciences", "Other"
}

ALLOWED_SECTORS = {
    "Academic", "Industry", "Government", "Healthcare", "Nonprofit", "Independent"
}


def load_hierarchy_info(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    parsed = {}
    errors = {}

    for paper_id, entry in cache.items():
        raw = entry.get("hierarchy_info", "")

        try:
            parsed[paper_id] = json.loads(raw)
        except json.JSONDecodeError as e:
            errors[paper_id] = str(e)

    return cache, parsed, errors


def sanity_check(cache_path):
    cache, parsed, json_errors = load_hierarchy_info(cache_path)

    problems = defaultdict(list)

    field_counter = Counter()
    research_area_counter = Counter()
    topic_family_counter = Counter()
    method_family_counter = Counter()
    method_category_counter = Counter()
    problem_counter = Counter()

    repeated_label = 0
    empty_affiliation_errors = 0
    invalid_sector_errors = 0
    missing_field_errors = 0

    for paper_id, data in parsed.items():
        base = cache[paper_id].get("base_info", {})
        original_affiliations = base.get("affiliations", "")

        # error of data not being a dictionary
        if not isinstance(data, dict):
            problems[paper_id].append(
                f"Expected dict but got {type(data).__name__}"
            )
            problem_counter[f"Expected dictionary, but got {type(data).__name__}"] += 1
            continue

        affiliations = data.get("affiliations")
        field_path = data.get("field_of_study_path", {})
        method_path = data.get("method_path", {})

        # 1. Required top-level keys
        for key in ["affiliations", "field_of_study_path", "method_path"]:
            if key not in data:
                problems[paper_id].append(f"Missing top-level key: {key}")
                problem_counter["Missing top-level key"] += 1

        # 2. Affiliation checks
        if not original_affiliations and affiliations != []:
            problems[paper_id].append("Affiliations should be [] because original affiliations are empty")
            empty_affiliation_errors += 1
            problem_counter["Affiliations non-empty, but gold data is "] += 1

        if isinstance(affiliations, list):
            for aff in affiliations:
                if not aff.get("organization"):
                    problems[paper_id].append("Affiliation has empty organization")
                    problem_counter["Affiliation has empty organization"] += 1
                if not aff.get("country"):
                    problem_counter["Affiliation has empty country"] += 1

                sector = aff.get("sector")
                if sector and sector not in ALLOWED_SECTORS:
                    problems[paper_id].append(f"Invalid sector: {sector}")
                    invalid_sector_errors += 1
                    problem_counter["Invalid sector"] += 1

        # 3. Field path checks
        field = field_path.get("field")
        research_area = field_path.get("research_area")
        topic_family = field_path.get("topic_family")
        specific_topic = field_path.get("specific_topic")

        if not field:
            problems[paper_id].append("Missing field")
            missing_field_errors += 1
            problem_counter["Missing field"] += 1

        if field:
            field_counter[field] += 1
        if research_area:
            research_area_counter[research_area] += 1
        if topic_family:
            topic_family_counter[topic_family] += 1

        # 4. Method path checks
        method_family = method_path.get("method_family")
        method_category = method_path.get("method_category")
        specific_method = method_path.get("specific_method")

        if method_family:
            method_family_counter[method_family] += 1
        if method_category:
            method_category_counter[method_category] += 1

        # 5. Repetition checks
        labels = [
            field,
            research_area,
            topic_family,
            specific_topic,
        ]

        next_labels = [
            method_family,
            method_category,
            specific_method
        ]

        labels = [x.lower().strip() for x in labels if isinstance(x, str)]
        next_labels = [x.lower().strip() for x in next_labels if isinstance(x, str)]

        if len(labels) != len(set(labels)):
            problems[paper_id].append("Repeated label in hierarchy")
            repeated_label+=1
            problem_counter["Repeated label in hierarchy field_of_study_path"] += 1
        if len(next_labels) != len(set(next_labels)):
            repeated_label+=1
            problem_counter["Repeated label in hierarchy method_path"] += 1

        # 6. Overly long labels
        for label in labels:
            if len(label.split()) > 8:
                problems[paper_id].append(f"Possibly too long label: {label}")
                problem_counter["Possibly too long label"] += 1

    print("\n===== SANITY CHECK REPORT =====")
    print(f"Total entries: {len(cache)}")
    print(f"Parsed successfully: {len(parsed)}")
    print(f"JSON errors: {len(json_errors)}")

    print("\nTop problems:")
    for k, v in problem_counter.most_common(20):
        print(f"{k}: {v}")

    print("\nStatistics:")
    print("\nTop fields:")
    for k, v in field_counter.most_common(20):
        print(f"{k}: {v}")

    print("\nTop research areas:")
    for k, v in research_area_counter.most_common(20):
        print(f"{k}: {v}")

    print("\nTop topic families:")
    for k, v in topic_family_counter.most_common(30):
        print(f"{k}: {v}")

    print("\nTop method families:")
    for k, v in method_family_counter.most_common(20):
        print(f"{k}: {v}")

    print("\nTop method categories:")
    for k, v in method_category_counter.most_common(30):
        print(f"{k}: {v}")

    return problems, json_errors
# --------------------------------- main program --------------------------------------------
if __name__ == "__main__":
    sanity_check("cache/hierarchy_cache_nano_4.json")