from aifx.site import render_document, render_fragment


def test_document_wraps_fragment_and_embeds_data():
    bundle = {"note": "</script><script>alert(1)</script>", "pairs": []}
    doc = render_document(bundle)
    assert doc.startswith("<!doctype html>")
    assert "<title>AIFX 為替予測</title>" in doc.split("</head>")[0]
    assert "__FX_DATA__" not in doc
    # The payload must not be able to terminate its <script> element.
    payload = doc.split('id="fx-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "alert(1)" in payload


def test_fragment_has_no_document_wrapper():
    frag = render_fragment({"pairs": []})
    assert "<html" not in frag and "<body" not in frag
    assert frag.lstrip().startswith("<title>")
