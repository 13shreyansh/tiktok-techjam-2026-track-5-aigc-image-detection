from scripts.acquire_openimages_validation_audit import parse_listing, rank_key


def test_listing_parser_extracts_public_object_contract():
    payload = b'''<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <NextContinuationToken>next token</NextContinuationToken>
      <Contents><Key>validation/abc.jpg</Key><LastModified>2018-01-01Z</LastModified><ETag>"deadbeef"</ETag><Size>12</Size></Contents>
      <Contents><Key>validation/readme.txt</Key><Size>4</Size></Contents>
    </ListBucketResult>'''
    rows, token = parse_listing(payload)
    assert token == "next token"
    assert rows == [{
        "key": "validation/abc.jpg",
        "bytes": 12,
        "etag": "deadbeef",
        "last_modified": "2018-01-01Z",
    }]


def test_seeded_key_rank_is_stable_and_seed_sensitive():
    assert rank_key("validation/a.jpg") == rank_key("validation/a.jpg")
    assert rank_key("validation/a.jpg") != rank_key("validation/a.jpg", 7)
