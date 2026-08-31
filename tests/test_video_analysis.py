from video_analysis import build_search_index, extract_quran_references, segment_topics


def test_extracts_quran_references_without_duplicates():
    assert extract_quran_references("Read 2:255, then return to 2:255 and 36:1.") == ["2:255", "36:1"]


def test_segments_transcript_into_searchable_topics():
    topics = segment_topics("First lesson. Second lesson about charity.")
    assert len(topics) == 2
    assert topics[0]["start_seconds"] == 0
    assert topics[1]["start_seconds"] == 30


def test_search_index_maps_transcript_tokens_to_positions():
    index = build_search_index("Arabic charity lecture", [])
    assert index["charity"] == [1]
