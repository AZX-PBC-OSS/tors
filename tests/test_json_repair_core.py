"""Port of json_repair's non-schema repair corpus into tors.

Provenance: json_repair (https://github.com/mangiucugna/json_repair) by
Stefano Baccianella, MIT license, commit
251d141786d0f6ff561f6ec04d90188a338e2470 (= 0.63.4).

Mapping: ``repair_json(raw)`` is ``tors.repair_json(raw)`` and
``repair_json(raw, return_objects=True)`` is ``tors.repair_json_loads(raw)``.
Every assertion carries a one-line comment citing its upstream test function.
Deliberately excluded (no tors equivalent): ``stream_stable``, ``logging=``,
pydantic inputs, ``json_fd``/``load``/``from_file``, CLI, monkeypatched
internals, private-helper direct calls, CountingParser counts, and
StringFileWrapper cases.
"""

from __future__ import annotations

import pytest

from tors import repair_json, repair_json_loads


def _assert_loads_both(raw: str, expected: object) -> None:
    """Assert the default and ``skip_json_loads=True`` spellings agree."""
    assert repair_json_loads(raw) == expected
    assert repair_json_loads(raw, skip_json_loads=True) == expected


class TestCorpusStrings:
    """Every str-out assertion from the six upstream corpus files."""

    def test_valid_json(self) -> None:
        assert (
            repair_json(  # upstream test_valid_json
                '{"name": "John", "age": 30, "city": "New York"}'
            )
            == '{"name": "John", "age": 30, "city": "New York"}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"employees":["John", "Anna", "Peter"]} '
            )
            == '{"employees": ["John", "Anna", "Peter"]}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"key": "value:value"}'
            )
            == '{"key": "value:value"}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"text": "The quick brown fox,"}'
            )
            == '{"text": "The quick brown fox,"}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"text": "The quick brown fox won\'t jump"}'
            )
            == '{"text": "The quick brown fox won\'t jump"}'
        )
        assert repair_json('{"key": ""') == '{"key": ""}'  # upstream test_valid_json
        assert (
            repair_json(  # upstream test_valid_json
                '{"key1": {"key2": [1, 2, 3]}}'
            )
            == '{"key1": {"key2": [1, 2, 3]}}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"key": 12345678901234567890}'
            )
            == '{"key": 12345678901234567890}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"key": "value\u263a"}'
            )
            == '{"key": "value\\u263a"}'
        )
        assert (
            repair_json(  # upstream test_valid_json
                '{"key": "value\\nvalue"}'
            )
            == '{"key": "value\\nvalue"}'
        )

    def test_multiple_jsons_str(self) -> None:
        assert repair_json("[]{}") == "[]"  # upstream test_multiple_jsons
        assert (
            repair_json(  # upstream test_multiple_jsons
                '[]{"key":"value"}'
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_multiple_jsons
                '{"key":"value"}[1,2,3,True]'
            )
            == '[{"key": "value"}, [1, 2, 3, true]]'
        )
        assert (
            repair_json(  # upstream test_multiple_jsons
                'lorem ```json {"key":"value"} ``` ipsum ```json [1,2,3,True] ``` 42'
            )
            == '[{"key": "value"}, [1, 2, 3, true]]'
        )
        assert (
            repair_json(  # upstream test_multiple_jsons
                '[{"key":"value"}][{"key":"value_after"}]'
            )
            == '[{"key": "value_after"}]'
        )

    def test_fenced_prose_str(self) -> None:
        raw_decision = (  # upstream test_parenthesized_prose_does_not_hijack_fenced_json
            "\n**Decision**: bla, bla (some clarification):\n\n"
            '```json\n{\n  "key": "value"\n}\n```\n'
        )
        assert repair_json(raw_decision) == '{"key": "value"}'
        raw_numbered = (  # upstream test_numbered_prose_line_does_not_hijack_fenced_json
            '\n(1) Keep this note in the explanation.\n\n```json\n{\n  "key": "value"\n}\n```\n'
        )
        assert repair_json(raw_numbered) == '{"key": "value"}'

    def test_skip_json_loads_str(self) -> None:
        assert (
            repair_json(  # upstream test_repair_json_skip_json_loads
                '{"key": true, "key2": false, "key3": null}', skip_json_loads=True
            )
            == '{"key": true, "key2": false, "key3": null}'
        )
        assert (
            repair_json(  # upstream test_repair_json_skip_json_loads
                '{"key": true, "key2": false, "key3": }', skip_json_loads=True
            )
            == '{"key": true, "key2": false, "key3": ""}'
        )

    def test_ensure_ascii_str(self) -> None:
        assert (
            repair_json(  # upstream test_ensure_ascii
                "{'test_\u4e2d\u56fd\u4eba_ascii':'\u7edf\u4e00\u7801'}", ensure_ascii=False
            )
            == '{"test_\u4e2d\u56fd\u4eba_ascii": "\u7edf\u4e00\u7801"}'
        )

    def test_recursion_payload_depth_500_raises(self) -> None:
        # upstream test_repair_json_normalizes_real_parser_recursion_error
        payload = ("{" + '"a": [' * 500) + "1" + ("]}" * 500)
        with pytest.raises(ValueError, match="recursion depth"):
            repair_json(payload)

    def test_parse_object_str(self) -> None:
        assert repair_json("   {  }   ") == "{}"  # upstream test_parse_object
        assert repair_json("{") == "{}"  # upstream test_parse_object
        assert repair_json("}") == ""  # upstream test_parse_object
        assert repair_json('{"') == "{}"  # upstream test_parse_object

    def test_parse_object_edge_cases_str(self) -> None:
        assert repair_json("{foo: [}") == '{"foo": []}'  # upstream test_parse_object_edge_cases
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"": "value"'
            )
            == '{"": "value"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": "v"alue"}'
            )
            == '{"key": "v\\"alue\\""}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"value_1": true, COMMENT "value_2": "data"}'
            )
            == '{"value_1": true, "value_2": "data"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"value_1": true, SHOULD_NOT_EXIST "value_2": "data" AAAA }'
            )
            == '{"value_1": true, "value_2": "data"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"" : true, "key2": "value2"}'
            )
            == '{"": true, "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{""answer"":[{""traits"":\'\'Female aged 60+\'\',""answer1"":""5""}]}'
            )
            == '{"answer": [{"traits": "Female aged 60+", "answer1": "5"}]}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{ "words": abcdef", "numbers": 12345", "words2": ghijkl" }'
            )
            == '{"words": "abcdef", "numbers": 12345, "words2": "ghijkl"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"number": 1,"reason": "According...""ans": "YES"}'
            )
            == '{"number": 1, "reason": "According...", "ans": "YES"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{ "a" : "{ b": {} }" }'
            )
            == '{"a": "{ b"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"b": "xxxxx" true}'
            )
            == '{"b": "xxxxx"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": "Lorem "ipsum" s,"}'
            )
            == '{"key": "Lorem \\"ipsum\\" s,"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"lorem": ipsum, sic, datum.",}'
            )
            == '{"lorem": "ipsum, sic, datum."}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"lorem": sic tamet. "ipsum": sic tamet, quick brown fox. "sic": ipsum}'
            )
            == '{"lorem": "sic tamet.", "ipsum": "sic tamet", "sic": "ipsum"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"lorem_ipsum": "sic tamet, quick brown fox. }'
            )
            == '{"lorem_ipsum": "sic tamet, quick brown fox."}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key":value, " key2":"value2" }'
            )
            == '{"key": "value", " key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key":value "key2":"value2" }'
            )
            == '{"key": "value", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                "{'text': 'words{words in brackets}more words'}"
            )
            == '{"text": "words{words in brackets}more words"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                "{text:words{words in brackets}}"
            )
            == '{"text": "words{words in brackets}"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                "{text:words{words in brackets}m}"
            )
            == '{"text": "words{words in brackets}m"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": "value, value2"```'
            )
            == '{"key": "value, value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": "value}```'
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                "{key:value,key2:value2}"
            )
            == '{"key": "value", "key2": "value2"}'
        )
        assert repair_json('{"key:"value"}') == '{"key": "value"}'  # upstream edge cases
        assert repair_json('{"key:value}') == '{"key": "value"}'  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '[{"lorem": {"ipsum": "sic"}, """" "lorem": {"ipsum": "sic"}]'
            )
            == '[{"lorem": {"ipsum": "sic"}}, {"lorem": {"ipsum": "sic"}}]'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{ "key": ["arrayvalue"], ["arrayvalue1"], ["arrayvalue2"], "key3": "value3" }'
            )
            == '{"key": ["arrayvalue", "arrayvalue1", "arrayvalue2"], "key3": "value3"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{ "key": [[1, 2, 3], "a", "b"], [[4, 5, 6], [7, 8, 9]] }'
            )
            == '{"key": [[1, 2, 3], "a", "b", [4, 5, 6], [7, 8, 9]]}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{ "key": ["arrayvalue"], "key3": "value3", ["arrayvalue1"] }'
            )
            == '{"key": ["arrayvalue"], "key3": "value3", "arrayvalue1": ""}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": "{\\\\"key\\\\\\":[\\"value\\\\\\"],\\"key2\\":"value2"}"}'
            )
            == '{"key": "{\\"key\\":[\\"value\\"],\\"key2\\":\\"value2\\"}"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"key": , "key2": "value2"}'
            )
            == '{"key": "", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '{"array":[{"key": "value"], "key2": "value2"}'
            )
            == '{"array": [{"key": "value"}], "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                '[{"key":"value"}},{"key":"value"}]'
            )
            == '[{"key": "value"}, {"key": "value"}]'
        )
        assert (
            repair_json(  # upstream test_parse_object_edge_cases
                "{'key': ['a':{'duplicated_key': 'duplicated_value', "
                "'duplicated_key': 'duplicated_value'}]}"
            )
            == '{"key": [{"a": {"duplicated_key": "duplicated_value"}}]}'
        )

    def test_parse_object_backslash_key_str(self) -> None:
        # upstream test_parse_object_preserves_backslash_escaped_keys
        raw = '{\\"key\\": \\"value\\"}'
        assert repair_json(raw, skip_json_loads=True) == '{"key": "value"}'

    def test_parse_object_merge_at_end_str(self) -> None:
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, "key2": "value2"}'
            )
            == '{"key": "value", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, "key2": }'
            )
            == '{"key": "value", "key2": ""}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, []'
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, ["abc"]'
            )
            == '[{"key": "value"}, ["abc"]]'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, {}'
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, "" : "value2"}'
            )
            == '{"key": "value", "": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key": "value"}, "key2" "value2"}'
            )
            == '{"key": "value", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_parse_object_merge_at_the_end
                '{"key1": "value1"}, "key2": "value2", "key3": "value3"}'
            )
            == '{"key1": "value1", "key2": "value2", "key3": "value3"}'
        )

    def test_parse_array_str(self) -> None:
        assert repair_json("[[1\n\n]") == "[[1]]"  # upstream test_parse_array

    def test_parse_array_edge_cases_str(self) -> None:
        assert repair_json("[{]") == "[]"  # upstream test_parse_array_edge_cases
        assert repair_json("[") == "[]"  # upstream test_parse_array_edge_cases
        assert repair_json('["') == "[]"  # upstream test_parse_array_edge_cases
        assert repair_json("]") == ""  # upstream test_parse_array_edge_cases
        assert repair_json("[1, 2, 3,") == "[1, 2, 3]"  # upstream edge cases
        assert repair_json("[1, 2, 3, ...]") == "[1, 2, 3]"  # upstream edge cases
        assert repair_json("[1, 2, ... , 3]") == "[1, 2, 3]"  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                "[1, 2, '...', 3]"
            )
            == '[1, 2, "...", 3]'
        )
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                "[true, false, null, ...]"
            )
            == "[true, false, null]"
        )
        assert repair_json('["a" "b" "c" 1') == '["a", "b", "c", 1]'  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"employees":["John", "Anna",'
            )
            == '{"employees": ["John", "Anna"]}'
        )
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"employees":["John", "Anna", "Peter'
            )
            == '{"employees": ["John", "Anna", "Peter"]}'
        )
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"key1": {"key2": [1, 2, 3'
            )
            == '{"key1": {"key2": [1, 2, 3]}}'
        )
        assert repair_json('{"key": ["value]}') == '{"key": ["value"]}'  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '["lorem "ipsum" sic"]'
            )
            == '["lorem \\"ipsum\\" sic"]'
        )
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"key1": ["value1", "value2"}, "key2": ["value3", "value4"]}'
            )
            == '{"key1": ["value1", "value2"], "key2": ["value3", "value4"]}'
        )
        raw_headers = (  # upstream test_parse_array_edge_cases
            '{"headers": ["A", "B", "C"], "rows": [["r1a", "r1b", "r1c"], '
            '["r2a", "r2b", "r2c"], "r3a", "r3b", "r3c"], '
            '["r4a", "r4b", "r4c"], ["r5a", "r5b", "r5c"]]}'
        )
        expected_headers = (
            '{"headers": ["A", "B", "C"], "rows": [["r1a", "r1b", "r1c"], '
            '["r2a", "r2b", "r2c"], ["r3a", "r3b", "r3c"], '
            '["r4a", "r4b", "r4c"], ["r5a", "r5b", "r5c"]]}'
        )
        assert repair_json(raw_headers) == expected_headers
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"key": ["value" "value1" "value2"]}'
            )
            == '{"key": ["value", "value1", "value2"]}'
        )
        assert repair_json(  # upstream test_parse_array_edge_cases
            '{"key": ["lorem "ipsum" dolor "sit" amet, "consectetur" ", '
            '"lorem "ipsum" dolor", "lorem"]}'
        ) == (
            '{"key": ["lorem \\"ipsum\\" dolor \\"sit\\" amet, \\"consectetur\\" ", '
            '"lorem \\"ipsum\\" dolor", "lorem"]}'
        )
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '{"k"e"y": "value"}'
            )
            == '{"k\\"e\\"y": "value"}'
        )
        assert repair_json('["key":"value"}]') == '[{"key": "value"}]'  # upstream edge cases
        assert repair_json('["key":"value"]') == '[{"key": "value"}]'  # upstream edge cases
        assert repair_json('[ "key":"value"]') == '[{"key": "value"}]'  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_array_edge_cases
                '[{"key": "value", "key'
            )
            == '[{"key": "value"}, ["key"]]'
        )
        assert repair_json("{'key1', 'key2'}") == '["key1", "key2"]'  # upstream edge cases

    def test_parse_array_missing_quotes_str(self) -> None:
        assert (
            repair_json(  # upstream test_parse_array_missing_quotes
                '["value1" value2", "value3"]'
            )
            == '["value1", "value2", "value3"]'
        )
        assert repair_json(  # upstream test_parse_array_missing_quotes
            '{"bad_one":["Lorem Ipsum", "consectetur" comment" ], '
            '"good_one":[ "elit", "sed", "tempor"]}'
        ) == (
            '{"bad_one": ["Lorem Ipsum", "consectetur", "comment"], '
            '"good_one": ["elit", "sed", "tempor"]}'
        )
        assert repair_json(  # upstream test_parse_array_missing_quotes
            '{"bad_one": ["Lorem Ipsum","consectetur" comment],"good_one": ["elit","sed","tempor"]}'
        ) == (
            '{"bad_one": ["Lorem Ipsum", "consectetur", "comment"], '
            '"good_one": ["elit", "sed", "tempor"]}'
        )

    def test_parse_string_basics_str(self) -> None:
        assert repair_json('"') == ""  # upstream test_parse_string
        assert repair_json("\n") == ""  # upstream test_parse_string
        assert repair_json(" ") == ""  # upstream test_parse_string
        assert repair_json("string") == ""  # upstream test_parse_string
        assert repair_json("stringbeforeobject {}") == "{}"  # upstream test_parse_string

    def test_missing_and_mixed_quotes_str(self) -> None:
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                "{'key': 'string', 'key2': false, \"key3\": null, \"key4\": unquoted}"
            )
            == '{"key": "string", "key2": false, "key3": null, "key4": "unquoted"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": "John", "age": 30, "city": "New York'
            )
            == '{"name": "John", "age": 30, "city": "New York"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": "John", "age": 30, city: "New York"}'
            )
            == '{"name": "John", "age": 30, "city": "New York"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": "John", "age": 30, "city": New York}'
            )
            == '{"name": "John", "age": 30, "city": "New York"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": John, "age": 30, "city": "New York"}'
            )
            == '{"name": "John", "age": 30, "city": "New York"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{\u201cslanted_delimiter\u201d: "value"}'
            )
            == '{"slanted_delimiter": "value"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": "John", "age": 30, "city": "New'
            )
            == '{"name": "John", "age": 30, "city": "New"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"name": "John", "age": 30, "city": "New York, "gender": "male"}'
            )
            == '{"name": "John", "age": 30, "city": "New York", "gender": "male"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '[{"key": "value", COMMENT "notes": "lorem "ipsum", sic." }]'
            )
            == '[{"key": "value", "notes": "lorem \\"ipsum\\", sic."}]'
        )
        assert repair_json('{"key": ""value"}') == '{"key": "value"}'  # upstream mixed quotes
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"key": "value", 5: "value"}'
            )
            == '{"key": "value", "5": "value"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"foo": "\\"bar\\""'
            )
            == '{"foo": "\\"bar\\""}'
        )
        assert repair_json('{"" key":"val"') == '{" key": "val"}'  # upstream mixed quotes
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"key": value "key2" : "value2" '
            )
            == '{"key": "value", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"key": "lorem ipsum ... "sic " tamet. ...}'
            )
            == '{"key": "lorem ipsum ... \\"sic \\" tamet. ..."}'
        )
        assert repair_json('{"key": value , }') == '{"key": "value"}'  # upstream mixed quotes
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"comment": "lorem, "ipsum" sic "tamet". To improve"}'
            )
            == '{"comment": "lorem, \\"ipsum\\" sic \\"tamet\\". To improve"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"key": "v"alu"e"} key:'
            )
            == '{"key": "v\\"alu\\"e"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '{"key": "v"alue", "key2": "value2"}'
            )
            == '{"key": "v\\"alue", "key2": "value2"}'
        )
        assert (
            repair_json(  # upstream test_missing_and_mixed_quotes
                '[{"key": "v"alu,e", "key2": "value2"}]'
            )
            == '[{"key": "v\\"alu,e", "key2": "value2"}]'
        )

    def test_escaping_str(self) -> None:
        assert repair_json("'\"'") == ""  # upstream test_escaping
        assert (
            repair_json(  # upstream test_escaping
                '{"key": \'string"\n\t\\le\'}'
            )
            == '{"key": "string\\"\\n\\t\\\\le"}'
        )
        assert repair_json(  # upstream test_escaping
            r'{"real_content": "Some string: Some other string \t Some string '
            r'<a href=\"https://domain.com\">Some link</a>"'
        ) == (
            r'{"real_content": "Some string: Some other string \t Some string '
            r'<a href=\"https://domain.com\">Some link</a>"}'
        )
        assert repair_json('{"key_1\n": "value"}') == '{"key_1": "value"}'  # upstream escaping
        assert repair_json('{"key\t_": "value"}') == '{"key\\t_": "value"}'  # upstream escaping
        assert (
            repair_json(  # upstream test_escaping
                "{\"key\": '\u0076\u0061\u006c\u0075\u0065'}"
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_escaping
                '{"key": "\\u0076\\u0061\\u006C\\u0075\\u0065"}', skip_json_loads=True
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_escaping
                """{"key": "valu\\'e"}"""
            )
            == """{"key": "valu'e"}"""
        )
        assert (
            repair_json(  # upstream test_escaping
                '{\'key\': "{\\"key\\": 1, \\"key2\\": 1}"}'
            )
            == '{"key": "{\\"key\\": 1, \\"key2\\": 1}"}'
        )

    def test_markdown_str(self) -> None:
        assert (
            repair_json(  # upstream test_markdown
                '{ "content": "[LINK]("https://google.com")" }'
            )
            == '{"content": "[LINK](\\"https://google.com\\")"}'
        )
        assert (
            repair_json(  # upstream test_markdown
                '{ "content": "[LINK](" }'
            )
            == '{"content": "[LINK]("}'
        )
        assert (
            repair_json(  # upstream test_markdown
                '{ "content": "[LINK](", "key": true }'
            )
            == '{"content": "[LINK](", "key": true}'
        )

    def test_leading_trailing_str(self) -> None:
        assert (
            repair_json(  # upstream test_leading_trailing_characters
                '````{ "key": "value" }```'
            )
            == '{"key": "value"}'
        )
        assert (
            repair_json(  # upstream test_leading_trailing_characters
                '{    "a": "",    "b": [ { "c": 1} ] \n}```'
            )
            == '{"a": "", "b": [{"c": 1}]}'
        )
        assert (
            repair_json(  # upstream test_leading_trailing_characters
                "Based on the information extracted, here is the filled JSON output: "
                "```json { 'a': 'b' } ```"
            )
            == '{"a": "b"}'
        )
        assert (
            repair_json(  # upstream test_leading_trailing_characters
                '\nThe next 64 elements are:\n```json\n{ "key": "value" }\n```'
            )
            == '{"key": "value"}'
        )

    def test_string_json_llm_block_str(self) -> None:
        assert repair_json('{"key": "``"') == '{"key": "``"}'  # upstream llm block
        assert repair_json('{"key": "```json"') == '{"key": "```json"}'  # upstream llm block
        assert (
            repair_json(  # upstream test_string_json_llm_block
                '{"key": "```json {"key": [{"key1": 1},{"key2": 2}]}```"}'
            )
            == '{"key": {"key": [{"key1": 1}, {"key2": 2}]}}'
        )
        assert (
            repair_json(  # upstream test_string_json_llm_block
                '{"response": "```json{}"'
            )
            == '{"response": "```json{}"}'
        )

    def test_inline_object_literals_str(self) -> None:
        # upstream test_parse_string_keeps_inline_object_literal_after_comma
        raw_after = (
            '{"x": "However, the provided user answer is {"blank_1": "music"}, '
            'which is not a plain string"}'
        )
        expected_after = (
            '{"x": "However, the provided user answer is {\\"blank_1\\": \\"music\\"}, '
            'which is not a plain string"}'
        )
        assert repair_json(raw_after) == expected_after
        assert repair_json(raw_after, skip_json_loads=True) == expected_after
        # upstream test_parse_string_keeps_inline_object_literal_before_next_member
        for raw, expected in [
            ('{"x": "a, {"k": 1}, "y": 2}', '{"x": "a, {\\"k\\": 1}", "y": 2}'),
            (
                '{"x": "a, {"k": {"n": 1}}, "y": 2}',
                '{"x": "a, {\\"k\\": {\\"n\\": 1}}", "y": 2}',
            ),
        ]:
            assert repair_json(raw) == expected
            assert repair_json(raw, skip_json_loads=True) == expected

    def test_repeated_backslashes_str(self) -> None:
        # upstream test_parse_string_preserves_repeated_escaped_backslashes_during_repair
        for raw, serialized in [
            (r'{"key": "a\\\\b\\\\c",}', r'{"key": "a\\\\b\\\\c"}'),
            (r'{"key": "1\\\\2\\\\3",}', r'{"key": "1\\\\2\\\\3"}'),
        ]:
            assert repair_json(raw) == serialized
            assert repair_json(raw, skip_json_loads=True) == serialized

    def test_empty_single_quoted_key_str(self) -> None:
        # upstream test_parse_string_empty_single_quoted_key
        assert repair_json("{'': 1}") == '{"": 1}'

    def test_boolean_literals_str(self) -> None:
        assert (
            repair_json(  # upstream test_parse_boolean_or_null
                '  {"key": true, "key2": false, "key3": null}'
            )
            == '{"key": true, "key2": false, "key3": null}'
        )
        assert (
            repair_json(  # upstream test_parse_boolean_or_null
                '{"key": TRUE, "key2": FALSE, "key3": Null}   '
            )
            == '{"key": true, "key2": false, "key3": null}'
        )

    def test_parse_number_edge_cases_str(self) -> None:
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                ' - { "test_key": ["test_value", "test_value2"] }'
            )
            == '{"test_key": ["test_value", "test_value2"]}'
        )
        assert repair_json('{"key": 1/3}') == '{"key": "1/3"}'  # upstream number edge cases
        assert repair_json('{"key": .25}') == '{"key": 0.25}'  # upstream number edge cases
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                '{"here": "now", "key": 1/3, "foo": "bar"}'
            )
            == '{"here": "now", "key": "1/3", "foo": "bar"}'
        )
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                '{"key": 12345/67890}'
            )
            == '{"key": "12345/67890"}'
        )
        assert repair_json("[105,12") == "[105, 12]"  # upstream number edge cases
        assert repair_json('{"key", 105,12,') == '{"key": "105,12"}'  # upstream edge cases
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                '{"key": 1/3, "foo": "bar"}'
            )
            == '{"key": "1/3", "foo": "bar"}'
        )
        assert repair_json('{"key": 10-20}') == '{"key": "10-20"}'  # upstream edge cases
        assert repair_json('{"key": 1.1.1}') == '{"key": "1.1.1"}'  # upstream edge cases
        assert repair_json("[- ") == "[]"  # upstream test_parse_number_edge_cases
        assert repair_json('{"key": 1. }') == '{"key": 1.0}'  # upstream number edge cases
        assert repair_json('{"key": 1e10 }') == '{"key": 10000000000.0}'  # upstream edge cases
        assert repair_json('{"key": 1e }') == '{"key": 1}'  # upstream number edge cases
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                '{"key": 1notanumber }'
            )
            == '{"key": "1notanumber"}'
        )
        assert (
            repair_json(  # upstream test_parse_number_edge_cases
                '{"rowId": 57eeeeb1-450b-482c-81b9-4be77e95dee2}'
            )
            == '{"rowId": "57eeeeb1-450b-482c-81b9-4be77e95dee2"}'
        )
        assert repair_json("[1, 2notanumber]") == '[1, "2notanumber"]'  # upstream edge cases

    def test_parse_comment_str(self) -> None:
        assert repair_json("/") == ""  # upstream test_parse_comment
        assert repair_json('/* comment */ {"key": "value"}')  # upstream test_parse_comment
        assert (
            repair_json(  # upstream test_parse_comment
                '{ "key": { "key2": "value2" // comment }, "key3": "value3" }'
            )
            == '{"key": {"key2": "value2"}}'
        )
        assert (
            repair_json(  # upstream test_parse_comment
                '{ "key": { "key2": "value2" // comment\n}, "key3": "value3" }'
            )
            == '{"key": {"key2": "value2"}, "key3": "value3"}'
        )
        assert (
            repair_json(  # upstream test_parse_comment
                '{ "key": { "key2": "value2" # comment }, "key3": "value3" }'
            )
            == '{"key": {"key2": "value2"}, "key3": "value3"}'
        )
        assert (
            repair_json(  # upstream test_parse_comment
                '{ "key": { "key2": "value2" /* comment */ }, "key3": "value3" }'
            )
            == '{"key": {"key2": "value2"}, "key3": "value3"}'
        )
        assert (
            repair_json(  # upstream test_parse_comment
                '[ "value", /* comment */ "value2" ]'
            )
            == '["value", "value2"]'
        )
        assert (
            repair_json(  # upstream test_parse_comment
                '{ "key": "value" /* comment'
            )
            == '{"key": "value"}'
        )


class TestCorpusLoads:
    """Every ``return_objects=True`` assertion from the six upstream files."""

    def test_repair_json_with_objects(self) -> None:
        assert repair_json_loads("[]") == []  # upstream test_repair_json_with_objects
        assert repair_json_loads("{}") == {}  # upstream test_repair_json_with_objects
        assert repair_json_loads(  # upstream test_repair_json_with_objects
            '{"key": true, "key2": false, "key3": null}'
        ) == {"key": True, "key2": False, "key3": None}
        assert repair_json_loads(  # upstream test_repair_json_with_objects
            '{"name": "John", "age": 30, "city": "New York"}'
        ) == {"name": "John", "age": 30, "city": "New York"}
        assert repair_json_loads("[1, 2, 3, 4]") == [1, 2, 3, 4]  # upstream objects
        assert repair_json_loads(  # upstream test_repair_json_with_objects
            '{"employees":["John", "Anna", "Peter"]} '
        ) == {"employees": ["John", "Anna", "Peter"]}
        raw_fhir = """{
  "resourceType": "Bundle",
  "id": "1",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "1",
        "name": [
          {"use": "official", "family": "Corwin",
           "given": ["Keisha", "Sunny"], "prefix": ["Mrs."]},
          {"use": "maiden", "family": "Goodwin",
           "given": ["Keisha", "Sunny"], "prefix": ["Mrs."]}
        ]
      }
    }
  ]
}"""  # upstream test_repair_json_with_objects (FHIR bundle)
        assert repair_json_loads(raw_fhir) == {
            "resourceType": "Bundle",
            "id": "1",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "1",
                        "name": [
                            {
                                "use": "official",
                                "family": "Corwin",
                                "given": ["Keisha", "Sunny"],
                                "prefix": ["Mrs."],
                            },
                            {
                                "use": "maiden",
                                "family": "Goodwin",
                                "given": ["Keisha", "Sunny"],
                                "prefix": ["Mrs."],
                            },
                        ],
                    }
                }
            ],
        }
        assert repair_json_loads(  # upstream test_repair_json_with_objects
            '{\n"html": "<h3 id="aaa">Waarom meer dan 200 Technical Experts - '
            '"Passie voor techniek"?</h3>"}'
        ) == {
            "html": '<h3 id="aaa">Waarom meer dan 200 Technical Experts - '
            '"Passie voor techniek"?</h3>'
        }
        raw_quotes = """[
  {"foo": "Foo bar baz", "tag": "#foo-bar-baz"},
  {"foo": "foo bar "foobar" foo bar baz.", "tag": "#foo-bar-foobar"}
]"""  # upstream test_repair_json_with_objects
        assert repair_json_loads(raw_quotes) == [
            {"foo": "Foo bar baz", "tag": "#foo-bar-baz"},
            {"foo": 'foo bar "foobar" foo bar baz.', "tag": "#foo-bar-foobar"},
        ]

    def test_multiple_jsons_loads(self) -> None:
        assert repair_json_loads(  # upstream test_multiple_jsons
            '{"key":"value"}, {"key":"value_after"}'
        ) == [{"key": "value"}, {"key": "value_after"}]

    def test_parenthesized_tuple_fenced_loads(self) -> None:
        # upstream test_parenthesized_tuple_still_parses_when_it_is_the_fenced_json_payload
        raw = "\nHere is the tuple payload:\n\n```json\n(1, 2)\n```\n"
        assert repair_json_loads(raw) == [1, 2]

    def test_skip_json_loads_loads(self) -> None:
        assert repair_json_loads(  # upstream test_repair_json_skip_json_loads
            '{"key": true, "key2": false, "key3": null}', skip_json_loads=True
        ) == {"key": True, "key2": False, "key3": None}

    def test_prefixed_valid_json_both_spellings(self) -> None:
        # upstream test_prefixed_valid_json_uses_value_fast_path_when_json_loads_is_skipped
        raw = (
            'Here is your JSON:\n{"text": "a\\n b c, floof: a\\n ... a b (c), '
            'floof: \\n a", "id": 8}'
        )
        expected = {"text": "a\n b c, floof: a\n ... a b (c), floof: \n a", "id": 8}
        assert repair_json_loads(raw) == expected
        assert repair_json_loads(raw, skip_json_loads=True) == expected

    def test_prefixed_trailing_text(self) -> None:
        # upstream test_prefixed_valid_json_with_trailing_text_uses_value_fast_path
        raw = 'Here is your JSON:\n{"text": "literal } and ]"}\nAdditional explanation.'
        assert repair_json_loads(raw) == {"text": "literal } and ]"}

    def test_tool_args_trailing_garbage_both_spellings(self) -> None:
        # upstream test_valid_json_with_trailing_garbage_preserves_string_content
        raw = (
            r"""{"tool_args": {"code": "# note\nconfig = {'type': 'object', """
            r"""'properties': {}, 'additionalProperties': True}"}}}"""
        )
        expected = {
            "tool_args": {
                "code": "# note\nconfig = {'type': 'object', 'properties': {}, "
                "'additionalProperties': True}",
            },
        }
        assert repair_json_loads(raw) == expected
        assert repair_json_loads(raw, skip_json_loads=True) == expected

    def test_truncated_prefix(self) -> None:
        # upstream test_prefixed_invalid_json_falls_back_to_repair_parser
        assert repair_json_loads('Here is your JSON: {"key": "value') == {"key": "value"}

    def test_parse_object_loads(self) -> None:
        assert repair_json_loads("{}") == {}  # upstream test_parse_object
        assert repair_json_loads(  # upstream test_parse_object
            '{ "key": "value", "key2": 1, "key3": True }'
        ) == {"key": "value", "key2": 1, "key3": True}
        assert repair_json_loads("{") == {}  # upstream test_parse_object
        assert repair_json_loads(  # upstream test_parse_object
            '{ "key": value, "key2": 1 "key3": null }'
        ) == {"key": "value", "key2": 1, "key3": None}

    def test_duplicate_key_and_set_forms(self) -> None:
        assert repair_json_loads(  # upstream test_parse_object_edge_cases
            '[{"b":"v2","b":"v2"}]', skip_json_loads=True
        ) == [{"b": "v2"}]
        assert repair_json_loads(  # upstream test_parse_object_edge_cases
            "{'item1', 'item2', 'item3'}", skip_json_loads=True
        ) == ["item1", "item2", "item3"]

    def test_backslash_key_and_empty_classifier(self) -> None:
        # upstream test_parse_object_preserves_backslash_escaped_keys (value only)
        assert repair_json_loads('{\\"key\\": \\"value\\"}', skip_json_loads=True) == {
            "key": "value"
        }
        # upstream test_parse_object_empty_object_classifier_keeps_objectish_inputs (values only)
        assert repair_json_loads("{:}", skip_json_loads=True) == {}
        assert repair_json_loads("{   }", skip_json_loads=True) == {}
        # upstream ..._keeps_array_fallback_for_backslash_noise (value shape only)
        assert isinstance(repair_json_loads(r"{foo\bar}", skip_json_loads=True), list)
        # upstream test_parse_object_empty_object_array_fallback_preserves_legacy_key_context
        assert repair_json_loads("[{5}s ", skip_json_loads=True) == [[5]]

    def test_parse_array_loads(self) -> None:
        assert repair_json_loads("[]") == []  # upstream test_parse_array
        assert repair_json_loads("[1, 2, 3, 4]") == [1, 2, 3, 4]  # upstream test_parse_array
        assert repair_json_loads("[") == []  # upstream test_parse_array

    def test_array_closes_before_member(self) -> None:
        raw = '{"outer": ["a", "b", "next": "value"}'  # upstream closes_before_object_member
        expected = {"outer": ["a", "b"], "next": "value"}
        assert repair_json_loads(raw) == expected
        # upstream test_parse_array_contextually_closes_in_strict_mode
        assert repair_json_loads(raw, strict=True) == expected

    def test_tuple_literals(self) -> None:
        assert repair_json_loads(  # upstream test_parse_array_python_tuple_literals
            '("a", "b", "c")'
        ) == ["a", "b", "c"]
        assert repair_json_loads("((1, 2), (3, 4))") == [[1, 2], [3, 4]]  # upstream tuples
        assert repair_json_loads(  # upstream test_parse_array_python_tuple_literals
            '{"coords": (1, 2), "ok": true}'
        ) == {"coords": [1, 2], "ok": True}
        assert repair_json_loads('{"empty": ()}') == {"empty": []}  # upstream tuples

    def test_tuple_booleans_nulls(self) -> None:
        assert repair_json_loads(  # upstream ..._accept_boolean_and_null_values
            "(true, false, null)", skip_json_loads=True
        ) == [True, False, None]
        assert repair_json_loads(  # upstream ..._accept_boolean_and_null_values
            "(True, False, None)", skip_json_loads=True
        ) == [True, False, None]
        assert repair_json_loads(  # upstream ..._accept_boolean_and_null_values
            '{"coords": (True, None)}', skip_json_loads=True
        ) == {"coords": [True, None]}

    def test_parenthesized_scalars(self) -> None:
        assert repair_json_loads("(1)") == 1  # upstream ..._scalar_keeps_scalar_shape
        assert repair_json_loads('("x")') == "x"  # upstream ..._scalar_keeps_scalar_shape
        assert repair_json_loads(  # upstream ..._scalar_keeps_scalar_shape
            '{"scalar_group": (1)}'
        ) == {"scalar_group": 1}
        assert repair_json_loads(  # upstream ..._scalar_keeps_scalar_shape
            '{"string_group": ("x")}'
        ) == {"string_group": "x"}

    def test_colon_prose_and_code_content(self) -> None:
        # upstream test_parse_string_keeps_colon_prose_inside_wrapped_valid_json
        _assert_loads_both(
            'Here\'s your JSON:\n{"stuff": [{"a": "foo", "blist": [{"text": '
            '"a\\n b c, floof: a\\n ... a b (c), floof: \\n a", "id": 8}]}]}',
            {
                "stuff": [
                    {
                        "a": "foo",
                        "blist": [
                            {
                                "text": "a\n b c, floof: a\n ... a b (c), floof: \n a",
                                "id": 8,
                            }
                        ],
                    }
                ]
            },
        )
        # upstream test_parse_string_keeps_code_like_content_inside_valid_json
        _assert_loads_both(
            r'{"command":"x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"}',
            {"command": "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"},
        )
        # upstream test_parse_string_keeps_pseudo_object_code_inside_valid_json
        _assert_loads_both(
            r'{"command":"x\nrollback: {registry: Registry, snapshot: EntitySnapshot}"}',
            {"command": "x\nrollback: {registry: Registry, snapshot: EntitySnapshot}"},
        )

    def test_literal_fenced_snippets(self) -> None:
        # upstream test_parse_string_keeps_literal_fenced_snippet_cases
        cases = [
            ('{\n"a": "\n```{}```\n",\n"b": "x",\n}', {"a": "\n```{}```", "b": "x"}),
            ('{\n"a": "\n```{}```\n"\n",\n"b": "x",\n}', {"a": '\n```{}```\n"', "b": "x"}),
            ('{\n"a": "\n```{}```\n"\n",\n\'b\': "x",\n}', {"a": '\n```{}```\n"', "b": "x"}),
            (
                '{\n"a": "\n```{}```\n"\n", // c\n"b": "x",\n}',
                {"a": '\n```{}```\n"', "b": "x"},
            ),
            ('{\n"a": "\n```{}```\n"\n",\n b: "x",\n}', {"a": '\n```{}```\n"', "b": "x"}),
            ('{"a":"```}```"a","b":"x"}', {"a": '```}```"a', "b": "x"}),
            ('{"a":"x}``` [1,2]\n","b":"y"}', {"a": "x}``` [1,2]", "b": "y"}),
            ('{"a":"x}``` [http://x]\n","b":"y"}', {"a": "x}``` [http://x]", "b": "y"}),
            ('{"a":"x}``` [foo[bar]\n","b":"y"}', {"a": "x}``` [foo[bar]", "b": "y"}),
            ('{"a":"x}``` [{\n","b":"y"}', {"a": "x}``` [{", "b": "y"}),
            ('{"a":"x}``` [foo, [bar]\n","b":"y"}', {"a": "x}``` [foo, [bar]", "b": "y"}),
            ('{"a":"x}``` [1,"z"]\n","b":"y"}', {"a": 'x}``` [1,"z"]', "b": "y"}),
            ('{"a":"x}``` [1, [2]]\n","b":"y"}', {"a": "x}``` [1, [2]]", "b": "y"}),
            ('{"a":"x}``` [1,[2],k:v]\n","b":"y"}', {"a": "x}``` [1,[2],k:v]", "b": "y"}),
            ('{"a":"x}``` (1,(2),k:v)\n","b":"y"}', {"a": "x}``` (1,(2),k:v)", "b": "y"}),
            ('{"a":"x}``` [1,2],\n","b":"y"}', {"a": "x}``` [1,2],", "b": "y"}),
            ('{"a":"x}``` // c\n [1,2]\n","b":"y"}', {"a": "x}``` // c\n [1,2]", "b": "y"}),
            ('{"a":"x}``` // c\n [1,2],\n","b":"y"}', {"a": "x}``` // c\n [1,2],", "b": "y"}),
            (
                '{\n"a": "\n```c\nint main() {\n}\n```\n'
                'Implementation: "xxx", xxx\n",\n"b": "x",\n}',
                {"a": '\n```c\nint main() {\n}\n```\nImplementation: "xxx", xxx', "b": "x"},
            ),
        ]
        for raw, expected in cases:
            _assert_loads_both(raw, expected)

    def test_stray_quote_and_smart_quotes(self) -> None:
        # upstream test_parse_string_stray_quote_line_before_trailing_comma_drops_stray_quote
        _assert_loads_both('{"a": "hello\n"\n",}', {"a": "hello"})
        # upstream ..._before_trailing_comma_at_eof_drops_stray_quote
        _assert_loads_both('{"a": "hello\n"\n",', {"a": "hello"})
        # upstream test_parse_string_keeps_multiline_curly_quoted_prose_after_comma
        _assert_loads_both(
            '{"x": "a,\n \u201cterm\u201d: explanation", "y": 2}',
            {"x": "a,\n \u201cterm\u201d: explanation", "y": 2},
        )
        # upstream test_parse_string_keeps_low_smart_quote_span_closed_by_ascii_quote
        _assert_loads_both(
            '{"text": "despre \u201eautocritic\u0103" \u0219i autocompasiune"}',
            {"text": 'despre \u201eautocritic\u0103" \u0219i autocompasiune'},
        )
        # upstream test_parse_string_keeps_low_smart_quote_span_closed_by_unicode_quote
        _assert_loads_both(
            '{"text": "despre \u201eautocritic\u0103\u201d \u0219i autocompasiune"}',
            {"text": "despre \u201eautocritic\u0103\u201d \u0219i autocompasiune"},
        )
        # upstream ..._closed_by_escaped_ascii_quote
        _assert_loads_both(
            '{"text": "aplica\u021bie \u201esham\\"), a f\u0103cut"}',
            {"text": 'aplica\u021bie \u201esham"), a f\u0103cut'},
        )
        # upstream test_parse_string_escaped_low_smart_quote_does_not_open_inner_span
        _assert_loads_both(
            '{"text": "a \\\u201e b", "y": 1}',
            {"text": "a \u201e b", "y": 1},
        )

    def test_leading_quoted_phrase_and_redundant_quote(self) -> None:
        # upstream test_parse_string_preserves_leading_quoted_phrase
        for raw, expected in [
            ('{"title": ""hello" world"}', {"title": '"hello" world'}),
            ('{"title": ""hello" world", "b": "y"}', {"title": '"hello" world', "b": "y"}),
        ]:
            assert repair_json_loads(raw) == expected
            assert repair_json_loads(raw, strict=True) == expected
        # upstream test_parse_string_removes_redundant_leading_quote
        assert repair_json_loads('{"key": ""value"}') == {"key": "value"}
        assert repair_json_loads('{"key": ""value"}', strict=True) == {"key": "value"}

    def test_bare_member_recovery(self) -> None:
        # upstream test_parse_string_keeps_bare_member_recovery_for_explicit_and_unclosed_values
        for raw, expected in [
            ('{"a": "first, b: "second"}', {"a": "first", "b": "second"}),
            ('{"a": "first, b: 1}', {"a": "first", "b": 1}),
            ('{"a": "first, b: true}', {"a": "first", "b": True}),
            ('{"a": "first, b: [1]}', {"a": "first", "b": [1]}),
            ('{"a": "first, b: prose}', {"a": "first", "b": "prose"}),
        ]:
            assert repair_json_loads(raw, skip_json_loads=True) == expected

    def test_regex_character_classes(self) -> None:
        # upstream test_parse_string_keeps_bare_quotes_inside_regex_character_classes
        expected = {
            "results": [
                {"regex": r"""^\s*path\(\s*['"]([^'"]+)['"]\s*,"""},
                {"regex": r"""^\s*re_path\(\s*[^'"]+['"]\s*,"""},
            ]
        }
        raw = """{
        "results": [
            {"regex": "^\\s*path\\(\\s*['\\"]([^'\\"]+)['\\"]\\s*,"},
            {"regex": "^\\s*re_path\\(\\s*[^'\\"]+['\\"]\\s*,"}
        ]
    }"""
        assert repair_json_loads(raw, skip_json_loads=True) == expected
        assert repair_json_loads(f"```json\n{raw}\n```", skip_json_loads=True) == expected

    def test_regular_members(self) -> None:
        # upstream test_parse_string_still_closes_regular_object_members_after_quoted_values
        assert repair_json_loads('{"first": "value", "second": "next"}', skip_json_loads=True) == {
            "first": "value",
            "second": "next",
        }

    def test_boolean_null_top_level(self) -> None:
        assert repair_json_loads("True") == ""  # upstream test_parse_boolean_or_null
        assert repair_json_loads("False") == ""  # upstream test_parse_boolean_or_null
        assert repair_json_loads("Null") == ""  # upstream test_parse_boolean_or_null
        assert repair_json_loads("true") is True  # upstream test_parse_boolean_or_null
        assert repair_json_loads("false") is False  # upstream test_parse_boolean_or_null
        assert repair_json_loads("null") is None  # upstream test_parse_boolean_or_null

    def test_python_none_and_boundaries(self) -> None:
        # upstream test_parse_literals_require_boundaries_and_support_python_none
        assert repair_json_loads('{"value": None }') == {"value": None}
        assert repair_json_loads("[None, TRUE, false, Null]") == [  # upstream literals
            None,
            True,
            False,
            None,
        ]
        assert repair_json_loads("[None") == [None]  # upstream literals
        assert repair_json_loads('{"value": "None"}') == {"value": "None"}  # upstream literals
        assert repair_json_loads('{"value": none}') == {"value": None}  # upstream literals
        assert repair_json_loads('{"value": NONE}') == {"value": None}  # upstream literals
        assert repair_json_loads(  # upstream literals
            '{"value": trueblue}'
        ) == {"value": "trueblue"}
        assert repair_json_loads(  # upstream literals
            '{"value": falsehood}'
        ) == {"value": "falsehood"}
        assert repair_json_loads(  # upstream literals
            '{"value": nullify}'
        ) == {"value": "nullify"}
        assert repair_json_loads(  # upstream literals
            '{"value": NoneType}'
        ) == {"value": "NoneType"}

    def test_far_quote_payloads(self) -> None:
        # upstream test_parse_string_far_quote_comma_payload_keeps_existing_repair_shape
        raw_comma = '{"a": "' + ("x," * 10_000) + '" tail'
        assert repair_json_loads(raw_comma, skip_json_loads=True) == {"a": "x," * 10_000}
        # upstream test_parse_string_far_quote_brace_payload_keeps_existing_repair_shape
        raw_brace = '{"a": "' + ("x}" * 5_000) + '" tail'
        assert repair_json_loads(raw_brace, skip_json_loads=True) == {"a": "x}" * 5_000}

    def test_escaped_braces_and_latex(self) -> None:
        # upstream test_parse_string_preserves_escaped_braces_after_comma_group
        assert repair_json_loads(r'{ "key": "\\{1,2\\} \\{3\\}" }', skip_json_loads=True) == {
            "key": r"\{1,2\} \{3\}"
        }
        # upstream test_parse_string_preserves_latex_command_after_bracketed_comma
        assert repair_json_loads(
            r'{ "key": "x [0,2] f(-\\frac{3}{4})" }', skip_json_loads=True
        ) == {"key": r"x [0,2] f(-\frac{3}{4})"}
        # upstream test_parse_string_keeps_latex_braces_before_inline_object_literal
        assert repair_json_loads(
            r'{ "llm_reason": "curve $C: \\frac{x^2}{m}$ answer is {"blank_1": "5"}, '
            r'correct is 4.", "llm_answer": "4" }',
            skip_json_loads=True,
        ) == {
            "llm_reason": r'curve $C: \frac{x^2}{m}$ answer is {"blank_1": "5"}, correct is 4.',
            "llm_answer": "4",
        }

    def test_missing_quotes_fragment(self) -> None:
        # upstream test_parse_string_missing_quotes_object_value_stops_at_quote_fragment
        assert repair_json_loads('{0:a"0"', skip_json_loads=True) == {"0": "a"}

    def test_brace_heuristics(self) -> None:
        # upstream test_parse_string_object_value_brace_heuristics
        for raw, expected in [
            ('{"key": "value}\\\\\\"more"}', {"key": 'value}"more'}),
            ('{"key": "value} "tail}', {"key": "value} "}),
            ('{"key": "value} "tail" more}', {"key": 'value} "tail" more'}),
            ('{"key": "value} key2: value2}', {"key": "value"}),
        ]:
            assert repair_json_loads(raw, skip_json_loads=True) == expected

    def test_repeated_backslashes_loads(self) -> None:
        # upstream test_parse_string_preserves_repeated_escaped_backslashes_during_repair
        for raw, expected in [
            (r'{"key": "a\\\\b\\\\c",}', r"a\\b\\c"),
            (r'{"key": "1\\\\2\\\\3",}', r"1\\2\\3"),
        ]:
            assert repair_json_loads(raw) == {"key": expected}
            assert repair_json_loads(raw, skip_json_loads=True) == {"key": expected}

    def test_parse_number_loads(self) -> None:
        assert repair_json_loads("1") == 1  # upstream test_parse_number
        assert repair_json_loads("1.2") == 1.2  # upstream test_parse_number
        # upstream test_parse_number (underscore literals)
        assert repair_json_loads('{"value": 82_461_110}') == {"value": 82461110}
        assert repair_json_loads('{"value": 1_234.5_6}') == {"value": 1234.56}  # upstream numbers

    def test_comment_loads(self) -> None:
        # upstream test_line_comment_brackets_do_not_trigger_empty_object_array_fallback (value)
        assert repair_json_loads("{\n// comment ]\n}", skip_json_loads=True) == {}
        # upstream test_block_comment_brackets_do_not_trigger_empty_object_array_fallback (value)
        assert repair_json_loads("{/* comment ] */}", skip_json_loads=True) == {}
        # upstream test_line_comment_brackets_do_not_close_array_items
        assert repair_json_loads(
            '{\n"Changes": [\n//object a\n{"Action": "1"},\n//object b ]\n'
            '{"Action": "2"},\n//object c ]\n{"Action": "3"}\n]\n}',
            skip_json_loads=True,
        ) == {"Changes": [{"Action": "1"}, {"Action": "2"}, {"Action": "3"}]}
        # upstream test_parse_many_top_level_comments_without_recursion_error (value only)
        assert repair_json_loads(
            ("# comment\n" * 600) + '{"key": "value"}', skip_json_loads=True
        ) == {"key": "value"}


class TestStrictMode:
    """Every case from upstream test_strict_mode.py with its exact match pattern."""

    def test_rejects_multiple_top_level_values(self) -> None:
        # upstream test_strict_rejects_multiple_top_level_values
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json('{"key":"value"}["value"]', strict=True)

    def test_rejects_comma_separated_same_shape_top_level_objects(self) -> None:
        # upstream test_strict_rejects_comma_separated_same_shape_top_level_objects
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json('{"key":"value"}, {"key":"value_after"}', strict=True)

    def test_rejects_adjacent_same_shape_top_level_objects(self) -> None:
        # upstream test_strict_rejects_adjacent_same_shape_top_level_objects
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json('{"key":"value"}{"key":"value_after"}', strict=True)

    def test_rejects_adjacent_same_shape_top_level_arrays(self) -> None:
        # upstream test_strict_rejects_adjacent_same_shape_top_level_arrays
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json("[1][2]", strict=True)

    @pytest.mark.parametrize(
        "payload",
        ['{"key":"value"}[]', '{"key":"value"}{}', "[]{}", "{}[]", "[1]{}"],
    )
    def test_rejects_falsy_top_level_values(self, payload: str) -> None:
        # upstream test_strict_rejects_falsy_top_level_values
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json(payload, strict=True)

    def test_duplicate_keys_inside_array(self) -> None:
        # upstream test_strict_duplicate_keys_inside_array
        with pytest.raises(ValueError, match="Duplicate key found"):
            repair_json('[{"key": "first", "key": "second"}]', strict=True, skip_json_loads=True)

    def test_rejects_empty_keys(self) -> None:
        # upstream test_strict_rejects_empty_keys
        with pytest.raises(ValueError, match="Empty key found"):
            repair_json('{"" : "value"}', strict=True, skip_json_loads=True)

    def test_requires_colon_between_key_and_value(self) -> None:
        # upstream test_strict_requires_colon_between_key_and_value
        with pytest.raises(ValueError, match="Missing ':' after key"):
            repair_json('{"missing" "colon"}', strict=True)

    def test_rejects_empty_values(self) -> None:
        # upstream test_strict_rejects_empty_values
        with pytest.raises(ValueError, match="Parsed value is empty"):
            repair_json('{"key": , "key2": "value2"}', strict=True, skip_json_loads=True)

    def test_rejects_empty_object_with_extra_characters(self) -> None:
        # upstream test_strict_rejects_empty_object_with_extra_characters
        with pytest.raises(ValueError, match="Parsed object is empty"):
            repair_json('{"dangling"}', strict=True)

    def test_rejects_empty_escaped_object_with_extra_characters(self) -> None:
        # upstream test_strict_rejects_empty_escaped_object_with_extra_characters
        with pytest.raises(ValueError, match="Parsed object is empty"):
            repair_json('{\\"key\\": \\"value\\"}', strict=True, skip_json_loads=True)

    def test_detects_immediate_doubled_quotes(self) -> None:
        # upstream test_strict_detects_immediate_doubled_quotes
        with pytest.raises(ValueError, match=r"doubled quotes followed by another quote\.$"):
            repair_json('{"key": """"}', strict=True)

    def test_detects_doubled_quotes_followed_by_string(self) -> None:
        # upstream test_strict_detects_doubled_quotes_followed_by_string
        with pytest.raises(
            ValueError,
            match="doubled quotes followed by another quote while parsing a string",
        ):
            repair_json('{"key": "" "value"}', strict=True)
