import json
from collections import Counter, defaultdict
from idlelib.iomenu import errors

from data_set_enrichment import remove_babble, remove_comments_json_text, repair_fragmented_schema


ALLOWED_FIELDS = {
    "Computer Science", "Mathematics", "Engineering", "Medicine", "Biology",
    "Physics", "Economics", "Psychology", "Social Sciences", "Other"
}

ALLOWED_SECTORS = {
    "Academic", "Industry", "Government", "Healthcare", "Nonprofit", "Independent"
}
# helper functions
def contains_json_object(raw):
    return isinstance(raw, str) and "{" in raw and "}" in raw

def is_missing_metadata_babble(raw):
    if not isinstance(raw, str):
        return False

    text = raw.lower()

    patterns = [
        "i don't have the actual title",
        "i don’t have the actual title",
        "i need:",
        "please supply the missing metadata",
        "please provide the missing metadata",
        "to fill the schema correctly",
        "title, abstract, affiliations",
        "i cannot fill the schema",
        "i need the title",
        "i need the abstract",
    ]

    return any(pattern in text for pattern in patterns)


def load_hierarchy_info(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    parsed = {}
    errors = {}

    for paper_id, entry in cache.items():
        raw = entry.get("hierarchy_info", "")

        if is_missing_metadata_babble(raw) and not contains_json_object(raw):
            errors[paper_id] = "missing_title_abstract_babble"
            continue

        try:
            parsed[paper_id] = json.loads(raw)
        except json.JSONDecodeError as e:
            try:
                clean_metadata = remove_comments_json_text(raw)
                clean_metadata = remove_babble(clean_metadata)

                try:
                    parsed[paper_id] = json.loads(clean_metadata)

                except json.JSONDecodeError:
                    fragment_list = json.loads("[" + clean_metadata + "]")
                    parsed[paper_id] = repair_fragmented_schema(fragment_list)

            except (json.JSONDecodeError, TypeError):
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

    # 0. check if all ids are unique
    all_keys_unique = False
    keys_set = set(parsed.keys())
    if len(keys_set) == len(parsed):
        all_keys_unique = True

    for paper_id, data in parsed.items():

        # error of data not being a dictionary
        if not isinstance(data, dict):
            problems[paper_id].append(
                f"Expected dict but got {type(data).__name__}"
            )
            problem_counter[f"Expected data to be dictionary, but got {type(data).__name__}"] += 1
            continue

        affiliations = data.get("affiliations")
        field_path = data.get("field_of_study_path", {})
        method_path = data.get("method_path", {})

        # errors of sub entries not being correct data type
        if not isinstance(method_path, dict):
            problems[paper_id].append(
                f"Expected dict but got {type(data).__name__}"
            )
            problem_counter[f"Expected method_path to be dictionary, but got {type(method_path).__name__}"] += 1
            continue

        if not isinstance(field_path, dict):
            problems[paper_id].append(
                f"Expected dict but got {type(data).__name__}"
            )
            problem_counter[f"Expected field_path to be dictionary, but got {type(field_path).__name__}"] += 1
            continue

        if not isinstance(affiliations, list):
            problems[paper_id].append(
                f"Expected dict but got {type(data).__name__}"
            )
            problem_counter[f"Expected affiliations to be list, but got {type(affiliations).__name__}"] += 1
            continue

        # 1. Required top-level keys missing
        for key in ["affiliations", "field_of_study_path", "method_path"]:
            if key not in data:
                problems[paper_id].append(f"Missing top-level key: {key}")
                problem_counter["Missing top-level key"] += 1

        # 2. Affiliation checks
        # if not original_affiliations and affiliations != []:
        #     problems[paper_id].append("Affiliations should be [] because original affiliations are empty")
        #     empty_affiliation_errors += 1
        #     problem_counter["Affiliations non-empty, but gold data is "] += 1

        if isinstance(affiliations, list):
            for aff in affiliations:
                if not aff.get("organization"):
                    problems[paper_id].append("Affiliation has empty organization")
                    problem_counter["Affiliation has empty organization"] += 1
                if not aff.get("country"):
                    problem_counter["Affiliation has empty country"] += 1

                sector = aff.get("sector")
                if isinstance(sector, list):
                    problems[paper_id].append(
                        f"Sector is list instead of string: {sector}"
                    )
                    problem_counter["Sector is list"] += 1
                    continue
                if sector and sector not in ALLOWED_SECTORS:
                    problems[paper_id].append(f"Invalid sector: {sector}")

                    problem_counter["Invalid sector"] += 1

        # 3. Field path checks
        field = field_path.get("field")
        research_area = field_path.get("research_area")
        topic_family = field_path.get("topic_family")
        specific_topic = field_path.get("specific_topic")

        if not field:
            problems[paper_id].append("Missing field")
            problem_counter["Missing field"] += 1
        if not research_area:
            problems[paper_id].append("Missing research_area")
            problem_counter["Missing research_area"] += 1
        if not topic_family:
            problems[paper_id].append("Missing topic_family")
            problem_counter["Missing topic_family"] += 1
        if not specific_topic:
            problems[paper_id].append("Missing specific_topic")
            problem_counter["Missing specific_topic"] += 1

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

        if not method_family:
            problems[paper_id].append("Missing field")
            problem_counter["Missing method_family"] += 1
        if not method_category:
            problems[paper_id].append("Missing research_area")
            problem_counter["Missing method_category"] += 1
        if not specific_method:
            problems[paper_id].append("Missing topic_family")
            problem_counter["Missing specific_method"] += 1

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

            problem_counter["Repeated label in hierarchy field_of_study_path"] += 1
        if len(next_labels) != len(set(next_labels)):

            problem_counter["Repeated label in hierarchy method_path"] += 1

        # 6. Overly long labels
        for label in labels:
            if len(label.split()) > 8:
                problems[paper_id].append(f"Possibly too long label: {label}")
                problem_counter["Possibly too long label, more than 8"] += 1

    # 7. Filter out Json problems because title/abstract missing
    json_error_missing_info = 0
    for key, error in json_errors.items():
        # expecting value is part of the errors message which is produced to the model returning
        # title and abstract empty
        if "missing_title_abstract_babble" in str(error):
            json_error_missing_info += 1

    print("\n===== SANITY CHECK REPORT =====")

    print(f"Total entries: {len(cache)}")
    print(f"Keys are all unique: {all_keys_unique}")
    print(f"Parsed successfully: {len(parsed)}")
    print(f"JSON errors: {len(json_errors)}")
    print(f"JSON errors because title/abstract missing: ~{json_error_missing_info}")
    print(f"real JSON errors: ~{len(json_errors)-json_error_missing_info}")

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
    sanity_check("cache/hierarchy_cache_nano_batch.json")