from public_knowledge import PublicKnowledgeCatalog, PublicRecord


def test_search_returns_only_published_matching_records() -> None:
    catalog = PublicKnowledgeCatalog([
        PublicRecord("event", "Lagos Quran workshop", "Public weekend event", "/events/1"),
        PublicRecord("event", "Draft workshop", "Private planning", "/events/2", published=False),
    ])
    results = catalog.search("quran workshop", kind="event")
    assert [item.url for item in results] == ["/events/1"]


def test_empty_queries_do_not_expose_catalog() -> None:
    catalog = PublicKnowledgeCatalog([PublicRecord("course", "Tajwid", "Course", "/courses/1")])
    assert catalog.search("   ") == []
