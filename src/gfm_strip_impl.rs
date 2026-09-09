//! GitHub-Flavored Markdown → plain text, the `to_text` normalizer behind
//! `tors.documents.to_text` (`src/documents_impl.rs` routes and converts
//! first; this module turns the markdown into text). Pure Rust, no pyo3.
//!
//! # What this is for, and the input contract that keeps it small
//!
//! The input is **our own engines' markdown output** — pdf_oxide's
//! `to_markdown_all`, anydoc's GFM serializer, html-to-markdown-rs, and
//! office_oxide's selectable backend — not arbitrary user-authored
//! markdown. That contract bounds the construct set to what the four
//! engines emit (probed 2026-09-09 against every crate in the tree):
//! ATX headings, bullet/ordered/task lists with indentation, GFM tables
//! with `\|`-escaped cell pipes, fenced code blocks, blockquotes
//! (including nested `> >` and quote/list mixtures), links/images,
//! `<scheme://…>` and `<user@…>` autolinks, `*`/`**`/`***`/`~~` emphasis,
//! equal-length backtick code spans, hard-break markers, and HTML
//! entities (anydoc escapes entity-shaped `&`; the others emit text
//! literally — which is why the entity policy is a caller-chosen flag,
//! set per engine by `documents_impl`). Exotic markdown outside that set
//! passes through unchanged rather than being mangled; the corpus tests
//! pin the actual per-format outputs.
//!
//! One measured engine fact shapes the emphasis rules: **no engine ever
//! emits `_` as an emphasis marker** — anydoc/office_oxide style with
//! `*`/`**`/`***`/`~~`, html-to-markdown-rs renders `<em>`/`<strong>` as
//! `*`/`**`, and pdf_oxide's bold detection emits `**`. anydoc is also the
//! only engine that escapes markdown-significant literal text (`\_`, `\*`,
//! `\|`, `\\`); the other three emit literal text bare. So a bare `_` in
//! engine markdown is always literal text, and the strip never treats it
//! as emphasis — CommonMark's intraword rule for `_` would still pair
//! ` _like_this_ ` shapes and corrupt pdf/anydoc literal text (measured
//! 2026-09: anydoc leaves whitespace-flanked `_` bare precisely because
//! CommonMark cannot pair it, and the strip went on to pair it anyway:
//! `a _ b and _ c` → `a  b and  c`). Bare `*`/`~` runs keep the intraword
//! guard instead: pdf_oxide/html/office emit literal text unescaped, so
//! `3*4*5` in a PDF must survive, at the documented price that an
//! intraword `<em>`'s `*` markers survive too.
//!
//! The fence contract, stated precisely (probed 2026-09-09): fences are
//! runs of 3+ backticks or tildes — ANY length, not only three. Both
//! fence-writing engines pick the fence as one longer than any marker run
//! in the code (html-to-markdown-rs's `longest_consecutive_backtick_run +
//! 1`, anydoc's `backtick_fence`, registry sources), so a `<pre>` holding
//! a ``` line renders behind a 4-backtick fence; and fences open in
//! containers — bare, inside blockquotes (`> &#96;&#96;&#96;`: both engines
//! prefix every quoted line with `> `), and inside list items
//! (`- &#96;&#96;&#96;`, closed at the marker-width indent; a nested
//! list's fence at indent 4). Tilde fences are accepted vocabulary though
//! no tors-lane engine emits one by default (html-to-markdown-rs's tilde
//! style is an option tors does not set; anydoc/office_oxide/pdf_oxide
//! write backticks or none — probed).
//!
//! # Shape of the output
//!
//! Structure survives as text conventions, not syntax: headings keep their
//! text (dropped `#`s), list items keep their indentation and ordered
//! numbering (an ordered `1. ` marker is already text, so it survives the
//! paragraph path unchanged), table rows keep cell boundaries as `" | "`
//! joins with the `|---|` separator rows dropped, code blocks keep their
//! content without fences, links become `label (url)` — collapsing to the
//! label alone when the label IS the url (pdf_oxide's `[url](url)` and
//! html-to-markdown-rs's `<url>` autolinks are the same link wearing two
//! syntaxes) — images keep their alt text, and paragraph breaks are
//! preserved. One deliberate non-goal: reproducing the source document's
//! exact whitespace — the engines' markdown has already reflowed it; the
//! strip only removes syntax.

/// Strip GFM syntax from `markdown`, producing plain text.
///
/// `unescape_entities` follows the engine that produced the markdown:
/// `true` for anydoc (which HTML-escapes entity-shaped `&` in its
/// markdown), `false` for pdf_oxide, html-to-markdown-rs, and office_oxide
/// (which emit text literally — un-escaping would corrupt text that
/// genuinely contains `&amp;`). It also gates the hard-break marker drop:
/// anydoc is the only engine that marks a hard break with a trailing bare
/// `\`, so the marker is only removed on that lane.
pub fn strip(markdown: &str, unescape_entities: bool) -> String {
    if markdown.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(markdown.len());
    // The OPENING fence, tracked while its content streams (see
    // [`OpenFence`]): a fence closes only on the opening run's own marker
    // char at the opening run's length or longer, in the opening fence's
    // own container context — anything else on a line is content, pushed
    // verbatim.
    let mut open_fence: Option<OpenFence> = None;
    // One pending paragraph break: blank lines are counted, not emitted,
    // so a run of blanks (or the flanking blanks of a dropped line, e.g.
    // around a removed horizontal rule) collapses to one break BY
    // CONSTRUCTION. This replaces a whole-string `\n\n\n` → `\n\n` pass
    // that rescanned the output after every replacement and — worse —
    // reached INSIDE code fences, collapsing the blank lines a code block
    // legitimately contains (measured 2026-09: html-to-markdown-rs keeps
    // two blank lines in a `<pre>` block verbatim; the old pass ate one).
    // Fence content is pushed verbatim below and never passes through
    // this accounting at all.
    let mut pending_break = false;
    for line in markdown.split('\n') {
        let trimmed_end = line.trim_end();
        // Fenced code blocks: the fences themselves go, the content stays
        // verbatim (a code block's interior is never markdown to strip,
        // and never break-accounted either). Openers and closers are
        // recognized AFTER container markers — the same composition
        // treatment tables got — so a fence inside a blockquote or a list
        // item (`> ``` ` / `- ``` `, never reachable by trim_start alone)
        // is still a fence; see [`split_fence_line`].
        let fence_line = split_fence_line(trimmed_end);
        if let Some(open) = &open_fence {
            if closes_fence(&fence_line, open) {
                open_fence = None;
                continue;
            }
            if pending_break {
                out.push('\n');
                pending_break = false;
            }
            // The content line keeps its code verbatim minus the container
            // machinery the engine prefixes it with (`> `, list-marker
            // indent) — machinery, not code — trailing whitespace
            // included, exactly as the line was written.
            out.push_str(strip_fence_content(line, open));
            out.push('\n');
            continue;
        }
        if let Some((marker, len)) = opening_fence_run(fence_line.content) {
            open_fence = Some(OpenFence {
                marker,
                len,
                quotes: fence_line.quotes,
                indent: fence_line.indent,
            });
            continue;
        }
        // Horizontal rules: `-` runs only (3+, alone). Every hr-emitting
        // engine spells the rule `---` — html-to-markdown-rs's TagKind::Hr
        // arm, anydoc's Block::Rule arm, office_oxide's ThematicBreak arm
        // (registry sources, 2026-09-09); pdf_oxide emits none — so `***`/
        // `___` are never rules in engine markdown: they are literal text
        // from the unescaped engines (measured: pdf-lane `_____` fill-in
        // blanks and `***` were dropped as rules), and anydoc ESCAPES a
        // literal `_`/`*`, so a bare run cannot be its output either.
        if is_horizontal_rule(trimmed_end) {
            continue;
        }
        // Table rows (indentation-tolerant: html-to-markdown-rs indents a
        // table nested in a list, and anydoc renders a table inside a
        // blockquote as `> | … |`). Separator rows and all-empty rows (the
        // empty header anydoc renders over a headerless table) are table
        // machinery, not text.
        if is_table_row(trimmed_end) {
            if let Some(row) = table_row_text(trimmed_end, unescape_entities) {
                push_line(&mut out, &mut pending_break, &row);
            }
            continue;
        }
        // Everything else: structural markers off, then the inline pass.
        // Heading/bullet/quote TEXT gets the same inline treatment as
        // paragraph text (measured 2026-09: a docx hyperlink inside a
        // heading or bullet arrived in to_text as raw `[label](url)`
        // because the structure arms returned the text as-is).
        let line_text = match classify(trimmed_end) {
            LineKind::MarkerOnly => None,
            LineKind::Heading { text } => Some(strip_inline(text, unescape_entities)),
            LineKind::Bullet { indent, text } => {
                if is_table_row(text) {
                    // a table as the first block of a list item (anydoc
                    // puts the row right after the marker: `- | a | b |`)
                    table_row_text(text, unescape_entities).map(|row| format!("{indent}{row}"))
                } else {
                    let text = drop_hardbreak_marker(text, unescape_entities);
                    Some(format!("{indent}{}", strip_inline(text, unescape_entities)))
                }
            }
            LineKind::Quoted { text } => {
                if is_table_row(text) {
                    table_row_text(text, unescape_entities)
                } else {
                    let text = drop_hardbreak_marker(text, unescape_entities);
                    Some(strip_inline(text, unescape_entities))
                }
            }
            LineKind::Paragraph { text } => {
                let text = drop_hardbreak_marker(text, unescape_entities);
                Some(strip_inline(text, unescape_entities))
            }
        };
        match line_text {
            Some(text) if !text.trim().is_empty() => {
                push_line(&mut out, &mut pending_break, &text);
            }
            // A blank line — or one whose entire content was markup —
            // counts as one paragraph break.
            Some(_) => pending_break = true,
            // Marker-only (an empty list item): no text, no break.
            None => {}
        }
    }
    // The output ends with at most one newline. (Trailing blank lines
    // inside a final code block are trimmed with everything else — a
    // trailing blank run carries no content, and interior blanks — the
    // ones that matter — were pushed verbatim.)
    while out.ends_with("\n\n") {
        out.pop();
    }
    // Whitespace-only input (an empty page's markdown, a lone newline) is
    // no text at all: the empty string, not a newline.
    if out.trim().is_empty() {
        return String::new();
    }
    out
}

/// Push one content line, flushing a single pending paragraph break
/// before it.
fn push_line(out: &mut String, pending_break: &mut bool, text: &str) {
    if *pending_break {
        out.push('\n');
        *pending_break = false;
    }
    out.push_str(text);
    out.push('\n');
}

/// The opening fence of a fenced code block, remembered while its content
/// streams. Three properties close it — and only their conjunction does:
///
/// - the same MARKER CHAR (`&#96;&#96;&#96;` fences close on backticks,
///   `~~~` fences on tildes; the old toggler flipped state on either, so a
///   tilde opener closed on a backtick run);
/// - a run AT LEAST the opening run's length — both fence-writing engines
///   pick the fence as one longer than any marker run in the code
///   (html-to-markdown-rs's `longest_consecutive_backtick_run + 1` and
///   anydoc's `backtick_fence`, registry sources), so an interior run is
///   always SHORTER: the measured corruption (2026-09-09, html
///   `<pre>a\n&#96;&#96;&#96;\nb</pre>` → a 4-backtick fence) was the
///   interior ``` line toggling the tracker off, code after it
///   emphasis-stripped, and post-closer text passing through verbatim with
///   its `*` markers intact;
/// - the OPENING fence's own container context (quote depth, effective
///   indent) — a fence inside a blockquote or list item opens and closes
///   there, so a same-shape run at another context is content.
struct OpenFence {
    marker: char,
    len: usize,
    quotes: usize,
    indent: usize,
}

/// One line parsed for the fence tracker: the container context (quote
/// depth and effective indent — the fields of [`OpenFence`]) and the
/// content left after the container machinery.
struct FenceLine<'a> {
    quotes: usize,
    indent: usize,
    content: &'a str,
}

/// Split one line into its fence container context and content: leading
/// whitespace, then blockquote markers (each `>` plus one optional space —
/// the same repeated strip [`classify`] performs), then ONE bullet marker
/// (`-`/`*`/`+` plus a space), whose two columns are recorded as indent —
/// the spelling a list continuation line uses for the same container
/// (probed 2026-09-09 over html-to-markdown-rs: a `<pre>` in a list item
/// renders `- &#96;&#96;&#96;` closed by a marker-width-indented fence, and
/// anydoc's render_list indents continuation lines by exactly the marker
/// width; a nested list's `  * &#96;&#96;&#96;` closes at indent 4). The
/// loop mirrors [`classify`]'s container walk so the two parsers can never
/// disagree about which markers are containers; the checkbox is
/// deliberately NOT folded in (no engine emits fenced code inside a task
/// item, and its columns would miscount the continuation width).
fn split_fence_line(line: &str) -> FenceLine<'_> {
    let mut quotes = 0;
    let mut indent = 0;
    let mut bullet = false;
    let mut rest = line;
    loop {
        let spaces = rest.len() - rest.trim_start().len();
        indent += spaces;
        rest = &rest[spaces..];
        if let Some(after) = rest.strip_prefix('>') {
            rest = after.strip_prefix(' ').unwrap_or(after);
            quotes += 1;
            continue;
        }
        if !bullet && let Some(after) = strip_list_marker(rest) {
            indent += 2;
            bullet = true;
            rest = after;
            continue;
        }
        return FenceLine {
            quotes,
            indent,
            content: rest,
        };
    }
}

/// `- `/`* `/`+ ` — one marker column and one space — for the fence
/// context walk only (see [`split_fence_line`]).
fn strip_list_marker(body: &str) -> Option<&str> {
    for marker in ["-", "*", "+"] {
        if let Some(rest) = body.strip_prefix(marker)
            && let Some(text) = rest.strip_prefix(' ')
        {
            return Some(text);
        }
    }
    None
}

/// A fence-opening run in post-container content: 3+ of one marker char,
/// obeying CommonMark's info-string rule for the backtick form — the info
/// string may not contain a backtick, else the line is an inline code
/// span's delimiter run, not a fence (anydoc's serializer emits exactly
/// that shape for code containing backticks: ``` &#96;&#96;&#96;x&#96;&#96;&#96; ```
/// is a code SPAN at line start, and the old toggler swallowed the rest of
/// the document as "fence content" — probed 2026-09-09 over the
/// code-span corpus). Tilde fences take any info string (CommonMark; the
/// tilde form is accepted vocabulary even though no tors-lane engine
/// emits it — html-to-markdown-rs's tilde style is an option tors does not
/// set, probed — so the strip's own contract, not an engine's, governs it).
fn opening_fence_run(content: &str) -> Option<(char, usize)> {
    for marker in ['`', '~'] {
        let mut run = 0;
        for c in content.chars() {
            if c != marker {
                break;
            }
            run += 1;
        }
        if run < 3 {
            continue;
        }
        let info = &content[run..];
        if marker == '`' && info.contains('`') {
            continue;
        }
        return Some((marker, run));
    }
    None
}

/// Whether this line CLOSES the open fence: a run of the opening fence's
/// marker char at least the opening length, with nothing after it but
/// whitespace (a closer carries no info string — ``` &#96;&#96;&#96;rust ```
/// inside an open block is content, not a close), in the opening fence's
/// OWN container context: same quote depth and the same effective indent.
/// The engines close exactly where they opened — `> &#96;&#96;&#96;` with
/// `> &#96;&#96;&#96;`, `- &#96;&#96;&#96;` with the marker-width indent,
/// a nested list's `  * &#96;&#96;&#96;` with indent 4 — so a run at any
/// other context (a BARE ``` inside a quoted or bulleted fence, an
/// indented one inside a bare fence) is content, never a close (probed
/// 2026-09-09: every interior line of a container fence carries the
/// container's markers).
fn closes_fence(line: &FenceLine<'_>, open: &OpenFence) -> bool {
    if line.quotes != open.quotes || line.indent != open.indent {
        return false;
    }
    let mut run = 0;
    for c in line.content.chars() {
        if c != open.marker {
            break;
        }
        run += 1;
    }
    run >= open.len && line.content[run..].trim().is_empty()
}

/// One content line of an open fence: the code verbatim, minus the
/// container machinery the engine prefixed it with — up to the opening
/// fence's indent in leading spaces, then up to its quote count of `>`
/// markers (each with one optional space), repeated while either budget
/// remains. The machinery is the engine's (`> quoted code` is quote plus
/// code; anydoc's empty quoted line is `>` alone), the rest is the code's
/// own bytes, trailing whitespace included. The indent trim is a byte
/// budget against a leading whitespace run that may hold multi-byte
/// whitespace chars (fuzz-found 2026-09-09: a two-space-plus-`\u{85}`
/// content line under an indent-3 fence sliced inside the `\u{85}`), so
/// it consumes only WHOLE whitespace chars while they fit — a char the
/// budget cannot fit is the code's own content, kept verbatim.
fn strip_fence_content<'a>(line: &'a str, open: &OpenFence) -> &'a str {
    let mut rest = line;
    let mut spaces = open.indent;
    let mut quotes = open.quotes;
    loop {
        let mut take = 0;
        for (_, ch) in rest.char_indices() {
            if !ch.is_whitespace() || take + ch.len_utf8() > spaces {
                break;
            }
            take += ch.len_utf8();
        }
        rest = &rest[take..];
        spaces -= take;
        if quotes == 0 {
            return rest;
        }
        let Some(after) = rest.strip_prefix('>') else {
            return rest;
        };
        rest = after.strip_prefix(' ').unwrap_or(after);
        quotes -= 1;
    }
}

/// A table row's text: cells split on UNESCAPED pipes (engines escape a
/// literal cell pipe as `\|` — anydoc's cell renderer and pdf_oxide's
/// table writer both do), each cell through the inline pass, joined with
/// `" | "`. Separator rows and all-empty rows are machinery — `None`.
fn table_row_text(row: &str, unescape_entities: bool) -> Option<String> {
    let trimmed = row.trim();
    if !is_table_row(trimmed) || is_table_separator(trimmed) {
        return None;
    }
    let cells = split_table_cells(trimmed);
    if cells.iter().all(|c| c.is_empty()) {
        return None;
    }
    Some(
        cells
            .iter()
            .map(|cell| strip_inline_cell(cell, unescape_entities))
            .collect::<Vec<_>>()
            .join(" | "),
    )
}

fn split_indent(line: &str) -> (&str, &str) {
    let indent_end = line.len() - line.trim_start().len();
    line.split_at(indent_end)
}

enum LineKind<'a> {
    /// A line that is only a list marker — an empty item, or the marker
    /// line html-to-markdown-rs emits around a list item whose sole
    /// content is a blockquote (`-` on its own line). No text to keep.
    MarkerOnly,
    Heading {
        text: &'a str,
    },
    Bullet {
        indent: &'a str,
        text: &'a str,
    },
    /// A blockquote's content: quote markers gone, text kept (a quoted
    /// paragraph is text, not structure — its indentation went with the
    /// markers).
    Quoted {
        text: &'a str,
    },
    /// A plain paragraph line: indentation kept (list continuation lines
    /// carry their nesting as spaces).
    Paragraph {
        text: &'a str,
    },
}

/// The structural classifiers, run as a loop over a line's leading
/// markup. Quote markers strip repeatedly (`> > text` — anydoc renders
/// each nesting level with its own `> `; html-to-markdown-rs does the
/// same); ONE heading-or-bullet marker strips after them (`> - item`,
/// `- > text`, `> ### quoted heading` — all probed 2026-09-09). A second
/// list marker never strips: `- - x` is not an emitted shape, and literal
/// leading `-` arrives escaped (`\-`) from anydoc or bare from pdf_oxide,
/// where stripping twice would eat real text.
fn classify(line: &str) -> LineKind<'_> {
    let mut rest = line;
    let mut indent = "";
    let mut heading = false;
    let mut bullet = false;
    let mut quoted = false;
    loop {
        let (ind, body) = split_indent(rest);
        if let Some(after) = body.strip_prefix('>') {
            rest = after.strip_prefix(' ').unwrap_or(after);
            // Before any list/heading marker, the quote's own indentation
            // is where the content sits; after one (`- > text`), the
            // marker's indent already carries the nesting.
            if !heading && !bullet {
                indent = ind;
            }
            quoted = true;
            continue;
        }
        // A marker with nothing after it (an empty list item —
        // html-to-markdown-rs emits the bare `-` line, anydoc `- ` which
        // line-trims to the same): no text at all. The one accepted
        // ambiguity: pdf_oxide emits literal text unescaped, so a PDF
        // whose literal line is a lone dash loses it — the GFM reading of
        // such a line is an empty item, and anydoc would have escaped a
        // literal one (`\-`).
        if body == "-" || body == "*" || body == "+" {
            return LineKind::MarkerOnly;
        }
        if !heading && !bullet {
            if let Some(text) = strip_heading_marker(body) {
                heading = true;
                rest = text;
                indent = ind;
                continue;
            }
            if let Some(text) = strip_bullet_marker(body) {
                bullet = true;
                rest = text;
                indent = ind;
                continue;
            }
        }
        return match (heading, bullet, quoted) {
            (true, _, _) => LineKind::Heading { text: body },
            (_, true, _) if body.is_empty() => LineKind::MarkerOnly,
            (_, true, _) => LineKind::Bullet { indent, text: body },
            (false, false, true) => LineKind::Quoted { text: body },
            // rest is the untouched original line here: a paragraph keeps
            // its leading indentation.
            (false, false, false) => LineKind::Paragraph { text: rest },
        };
    }
}

fn strip_heading_marker(body: &str) -> Option<&str> {
    let rest = strip_prefix_char_run(body, '#')?;
    rest.strip_prefix(' ')
}

fn strip_bullet_marker(body: &str) -> Option<&str> {
    for marker in ["-", "*", "+"] {
        if let Some(rest) = body.strip_prefix(marker)
            && let Some(text) = rest.strip_prefix(' ')
        {
            // Task-list checkboxes are markers too: "[x] done" -> "done".
            return Some(strip_checkbox(text));
        }
    }
    None
}

fn strip_checkbox(text: &str) -> &str {
    match text.strip_prefix("[ ] ") {
        Some(after) => after,
        None => match text
            .strip_prefix("[x] ")
            .or_else(|| text.strip_prefix("[X] "))
        {
            Some(after) => after,
            None => text,
        },
    }
}

fn strip_prefix_char_run(text: &str, ch: char) -> Option<&str> {
    let stripped = text.trim_start_matches(ch);
    (stripped != text).then_some(stripped)
}

/// A horizontal rule: 3+ hyphens, whitespace-run-interleaved or not —
/// `-` is the one marker any engine's rule render emits (see the call
/// site's probe), so `*`/`_` runs are text and excluded (a spaced
/// `* * *` rule — exotic, none of ours — survives as literal, at the
/// price of its stars pairing as emphasis in the inline pass; the
/// engine-output contract takes the dash rule for the text-preserving
/// side of that trade).
fn is_horizontal_rule(line: &str) -> bool {
    let mut markers = 0;
    for c in line.chars() {
        if c.is_whitespace() {
            continue;
        }
        if c != '-' {
            return false;
        }
        markers += 1;
    }
    markers >= 3
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.ends_with('|') && t.len() >= 2
}

/// A GFM table's delimiter row: every cell matches the delimiter grammar —
/// hyphens optionally flanked by colons (`:--`, `--:`, `:-:`, `---`) —
/// with the engines' own spelling as the floor: a bare dash or two is a
/// delimiter cell only when a colon shows alignment intent (`:-`, `-:`).
/// GFM's raw grammar accepts the lone `-`, but every engine writes `---`
/// (anydoc's `format_row`, html-to-markdown-rs's `| --- |`, pdf_oxide's
/// table writer), and a DATA row whose every cell is a bare dash — a
/// spreadsheet row of `-` placeholders — was dropped as machinery
/// (measured 2026-09-09: `| - | - |` vanished). One cell failing the
/// grammar makes the whole row data, not machinery.
fn is_table_separator(trimmed: &str) -> bool {
    let mut cells = 0;
    for cell in trimmed.trim().split('|') {
        let cell = cell.trim();
        if cell.is_empty() {
            continue;
        }
        let dashes = cell.matches('-').count();
        let colons = cell.matches(':').count();
        if dashes == 0 || dashes + colons != cell.chars().count() || (dashes < 3 && colons == 0) {
            return false;
        }
        cells += 1;
    }
    cells > 0
}

/// Split a table row's inner cell text on UNESCAPED pipes: a `|` preceded
/// by an odd backslash run is a literal, escaped pipe (both table-writing
/// engines emit `\|` for a cell containing `|`; an even run is a literal
/// backslash whose pipe is a real cell boundary). Cells are trimmed (the
/// row's own padding); empty cells survive INTERIOR (a column position —
/// the engines' empty spreadsheet cells) and drop from the edges (the
/// row's frame).
fn split_table_cells(line: &str) -> Vec<&str> {
    let t = line.trim();
    let inner = t.strip_prefix('|').unwrap_or(t);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    let mut cells: Vec<&str> = Vec::new();
    let mut start = 0;
    let mut backslashes = 0usize;
    for (at, c) in inner.char_indices() {
        match c {
            '\\' => backslashes += 1,
            '|' => {
                if backslashes.is_multiple_of(2) {
                    cells.push(&inner[start..at]);
                    start = at + 1;
                }
                backslashes = 0;
            }
            _ => backslashes = 0,
        }
    }
    cells.push(&inner[start..]);
    let mut cells: Vec<&str> = cells.iter().map(|c| c.trim()).collect();
    let first = cells
        .iter()
        .position(|c| !c.is_empty())
        .unwrap_or(cells.len());
    let last = cells
        .iter()
        .rposition(|c| !c.is_empty())
        .map_or(0, |i| i + 1);
    if first < last {
        cells.truncate(last);
        cells.drain(..first);
    } else {
        cells.clear();
    }
    cells
}

/// Drop a hard-break marker from the end of a line's raw text: anydoc
/// ends an intra-paragraph line with a bare `\` — an odd trailing
/// backslash run, exactly its own `trim_paragraph`/`ends_with_hard_break`
/// rule (literals are always escaped pairwise, `\\`, so odd means the
/// marker). The other engines mark hard breaks with trailing SPACES
/// (html-to-markdown-rs's Spaces style, office_oxide) — already gone via
/// `trim_end` — or not at all (pdf_oxide), so the drop runs only on the
/// anydoc lane, where an odd run is unambiguous; a pdf lane line ending
/// in a literal `\` (say `C:\`) passes through untouched.
fn drop_hardbreak_marker(text: &str, unescape_entities: bool) -> &str {
    if !unescape_entities {
        return text;
    }
    let run = text.chars().rev().take_while(|&c| c == '\\').count();
    if run % 2 == 1 {
        &text[..text.len() - 1]
    } else {
        text
    }
}

/// Inline transforms: code spans are lifted out first (their interiors are
/// literal, never markup), then — in table cells — anydoc's `<br>` cell
/// break, then images, links, emphasis, and the markdown backslash
/// escapes the engines emit for literal marker characters (`\*` → `*`) —
/// and — last — the optional entity un-escape, so entities inside the
/// restored code spans are left alone exactly as the engine wrote them.
fn strip_inline(line: &str, unescape_entities: bool) -> String {
    strip_inline_mode(line, unescape_entities, false)
}

/// The table-cell inline pass: the same pipeline plus anydoc's cell-break
/// `<br>` → space (a multi-paragraph or hard-broken cell renders as
/// `first<br>second` in its markdown; a literal `<br>` in source text
/// arrives backslash-escaped, `\<br>`, and is left alone).
fn strip_inline_cell(line: &str, unescape_entities: bool) -> String {
    strip_inline_mode(line, unescape_entities, true)
}

fn strip_inline_mode(line: &str, unescape_entities: bool, in_cell: bool) -> String {
    let mut code_spans: Vec<String> = Vec::new();
    let lifted = lift_code_spans(line, &mut code_spans);
    let unbroken = if in_cell {
        replace_cell_breaks(&lifted)
    } else {
        lifted
    };
    let unmarked = strip_emphasis_and_links(&unbroken);
    let unescaped = unescape_marker_escapes(&unmarked);
    let mut restored = restore_code_spans(&unescaped, &code_spans);
    if unescape_entities {
        restored = crate::html_impl::unescape(&restored).into_owned();
    }
    restored
}

/// Replace anydoc's unescaped `<br>` cell breaks with a space. Runs AFTER
/// code spans are lifted (a `<br>` inside a code span is code, not a
/// break) and BEFORE marker un-escaping (a literal `\<br>` is not a
/// break). `<br>` is ASCII, so byte-level matching cannot split a UTF-8
/// sequence.
fn replace_cell_breaks(line: &str) -> String {
    if !line.contains("<br>") {
        return line.to_string();
    }
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut at = 0;
    while let Some(found) = line[at..].find("<br>") {
        let found = at + found;
        let mut backslashes = 0;
        while found > backslashes && bytes[found - 1 - backslashes] == b'\\' {
            backslashes += 1;
        }
        if backslashes % 2 == 0 {
            out.push_str(&line[at..found]);
            out.push(' ');
        } else {
            out.push_str(&line[at..found + 4]);
        }
        at = found + 4;
    }
    out.push_str(&line[at..]);
    out
}

/// Remove the backslash before ASCII punctuation: the engines escape
/// markdown-significant characters in source text (`\*`, `\_`, `\|`), and
/// plain text wants the characters, not the escapes. Applied AFTER code
/// spans are lifted (a code span's escapes are literal) and BEFORE they are
/// restored; the alignment normalizer on the Python side applies the same
/// rule, so both modes agree on what the text IS.
fn unescape_marker_escapes(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.clone().next() {
                Some(next) if next.is_ascii_punctuation() => {
                    out.push(next);
                    chars.next();
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Replace code spans with `\u{0}<idx>\u{0}` sentinels. NUL never appears
/// in the engines' markdown text (they reject/escape control characters),
/// so it is an unambiguous placeholder.
///
/// A span opens with a backtick run of any length and closes with a run of
/// the SAME length (CommonMark pairs equal backtick strings) — anydoc's
/// serializer emits exactly that for code containing backticks (its
/// `backtick_fence` picks one longer than any run inside, plus CommonMark's
/// one-space pad when the content starts or ends with a backtick; measured
/// 2026-09-09 over an epub `<code>a`b</code>`: `` ``a`b`` `). Shorter and
/// longer runs inside the content never close it; an UNPAIRED opener run
/// is literal text (the anydoc escaped-backtick shape leaves exactly one).
fn lift_code_spans(line: &str, spans: &mut Vec<String>) -> String {
    let mut out = String::with_capacity(line.len());
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0;
    // Backslash-escape state: a `\`` is a LITERAL backtick (the engines
    // escape exactly this shape — anydoc emits `\`` for a backtick in
    // source text), never a code-span opener, and neither is the bare
    // backtick that follows it. Without this state the lifter pairs the
    // escaped backtick with a later bare one, swallows both, and strands
    // the backslash (measured 2026-09: `backticks \`inline` shaped text`
    // stripped to `backticks \inline shaped text`, losing the literal
    // text). The escaped char flows through verbatim;
    // `unescape_marker_escapes` (which runs after lifting) then renders
    // `\`` as the literal backtick the document carries.
    let mut escaped = false;
    while i < bytes.len() {
        let c = bytes[i];
        if escaped {
            out.push(c);
            escaped = false;
            i += 1;
            continue;
        }
        match c {
            '\\' => {
                out.push(c);
                escaped = true;
                i += 1;
            }
            '`' => {
                let run_len = run_length(&bytes, i, '`');
                if let Some(close) = find_equal_run(&bytes, i + run_len, run_len) {
                    let mut content: String = bytes[i + run_len..close].iter().collect();
                    // CommonMark's padding: one space dropped from each
                    // end when both are spaces and the content is not all
                    // spaces — anydoc adds exactly this pad for content
                    // that starts or ends with a backtick.
                    if content.starts_with(' ')
                        && content.ends_with(' ')
                        && !content.trim().is_empty()
                    {
                        content = content[1..content.len() - 1].to_string();
                    }
                    spans.push(content);
                    out.push('\u{0}');
                    out.push_str(&spans.len().to_string());
                    out.push('\u{0}');
                    i = close + run_len;
                    continue;
                }
                // Unpaired: the whole run is literal text.
                for _ in 0..run_len {
                    out.push('`');
                }
                i += run_len;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// The position of the backtick run of exactly `len` starting at or after
/// `from` (a run of any other length — shorter or longer — is not a
/// closing candidate).
fn find_equal_run(chars: &[char], from: usize, len: usize) -> Option<usize> {
    let mut i = from;
    while i + len <= chars.len() {
        let exact_run = (0..len).all(|k| chars[i + k] == '`')
            && (i == 0 || chars[i - 1] != '`')
            && (i + len == chars.len() || chars[i + len] != '`');
        if exact_run {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// The recursion-depth cap for [`strip_emphasis_and_links`]: the inline
/// pass recurses once per nesting level of a link label, an image label,
/// or emphasis content, and that recursion is otherwise unbounded.
/// Measured 2026-09-09 (PR #27 red-team follow-up): a docx whose
/// paragraph text is `[[[…]]()…]()` ~30,000 deep (~120 KB) drove it into
/// a stack overflow — SIGSEGV, exit -11, uncatchable in any binding (the
/// recursion runs on the caller's stack, whatever thread that is). 256 is
/// the document engines' own nesting convention, adopted for the same
/// role here: anydoc caps XML depth at 256, and office_oxide 0.1.10's
/// `MAX_NESTING_DEPTH` is 256 — empirically calibrated by its authors
/// against a dedicated 16 MB parse stack (their table: the release cliff
/// is 3,000-4,000 levels there, 512-1,024 on a bare 2 MiB thread). This
/// pass has no dedicated stack, so 256 carries the same 2x margin against
/// the 2 MiB figure while sitting far past any real engine output (no
/// engine nests labels more than 2-3 deep — the module docs' input
/// contract). Past the cap the remainder degrades to literal text (see
/// [`strip_emphasis_and_links_at_depth`]).
const MAX_INLINE_DEPTH: usize = 256;

/// `[label](url)` → `label (url)` — `label` alone when the label IS the
/// destination (pdf_oxide renders a bare URL as `[url](url)` and a bare
/// email as `[e](mailto:e)`; html-to-markdown-rs emits `<url>`/`<e>`
/// autolinks for the same links — the strip renders one shape for both);
/// `![alt](src)` → `alt`; `<url>` autolinks → the url itself; emphasis
/// marker pairs unwrapped (a link label's own content gets the same
/// treatment — anydoc labels carry styled runs). Nested brackets inside
/// labels arrive from no engine (real output nests 1-3 deep); the
/// pathological `[[[…]]()…]()` shape is pinned by the depth-cap test —
/// see [`MAX_INLINE_DEPTH`]. A `[` without a matching `](url)` passes
/// through.
fn strip_emphasis_and_links(line: &str) -> String {
    strip_emphasis_and_links_at_depth(line, 0)
}

/// [`strip_emphasis_and_links`]'s worker, carrying the recursion depth
/// (one level per unwrapped label/image/emphasis nest). At
/// [`MAX_INLINE_DEPTH`] the line is returned verbatim: the remaining
/// label is emitted as literal text — lossy (its own `[…](…)` machinery
/// stays) but non-crashing, the same degradation the measured engines
/// show on deep HTML.
fn strip_emphasis_and_links_at_depth(line: &str, depth: usize) -> String {
    if depth >= MAX_INLINE_DEPTH {
        return line.to_string();
    }
    let chars: Vec<char> = line.chars().collect();
    // Which characters are backslash-escaped (the engines escape literal
    // markers: anydoc's styled runs escape their own content — `*a \* b*`
    // — and its link labels escape `]`). An escaped marker is literal
    // text, never a delimiter (CommonMark: a delimiter run is a string of
    // UNESCAPED marker characters), so every scan below consults this.
    let mut escaped_at = vec![false; chars.len()];
    let mut esc = false;
    for (idx, c) in chars.iter().enumerate() {
        if esc {
            escaped_at[idx] = true;
            esc = false;
        } else if *c == '\\' {
            esc = true;
        }
    }
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if escaped_at[i] || c == '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        // Autolinks: html-to-markdown-rs (its `autolinks` option, ON by
        // default) renders an <a> whose text is its absolute-URI href —
        // or a mailto whose text is the bare address — as
        // `<scheme://…>` / `<user@…>`. Plain text wants the destination
        // itself, brackets dropped. Anything else in angle brackets is
        // not an autolink shape and passes through.
        if c == '<'
            && let Some((content, next)) = parse_autolink(&chars, i)
        {
            out.push_str(&content);
            i = next;
            continue;
        }
        // Images first: '!' immediately followed by a link form.
        if c == '!'
            && i + 1 < chars.len()
            && chars[i + 1] == '['
            && let Some((label, _url, next)) = parse_link(&chars, &escaped_at, i + 1)
        {
            // An image contributes its alt text only: the source is a
            // binary reference, noise for plain text.
            out.push_str(&strip_emphasis_and_links_at_depth(&label, depth + 1));
            i = next;
            continue;
        }
        if c == '['
            && let Some((label, url, next)) = parse_link(&chars, &escaped_at, i)
        {
            if url == label || url == format!("mailto:{label}") {
                out.push_str(&label);
            } else {
                out.push_str(&strip_emphasis_and_links_at_depth(&label, depth + 1));
                if !url.is_empty() {
                    out.push_str(" (");
                    out.push_str(&url);
                    out.push(')');
                }
            }
            i = next;
            continue;
        }
        // Emphasis: `*` and `~` runs of one to three (`*x*`, `**x**`,
        // `***x***`, `~~x~~`) open a span closed by the same run; unwrap
        // the markers, keep the content. `_` is deliberately absent — no
        // engine emits it as a marker, so a bare `_` is always literal
        // text (see the module docs). Literal `*`/`~` arrive ESCAPED from
        // anydoc (`\*`), so a bare run in its markdown is emphasis —
        // EXCEPT intraword runs (`3*4*5`), which this keeps literal:
        // pdf_oxide, html-to-markdown-rs, and office_oxide emit literal
        // text UNESCAPED, and intraword stars in a PDF are arithmetic,
        // not styling. The price (measured, accepted): an intraword
        // `<em>`'s markers survive as literal `*`s. A run that does not
        // pair — or is intraword-flanked — passes through WHOLE:
        // re-reading the tail of a disqualified `**` as a fresh `*` run
        // half-stripped it (measured 2026-09-09: anydoc/office_oxide
        // render mid-word bold runs as `foo**bar**baz`, which stripped to
        // `foo*bar*baz`, mangling the text twice over).
        if matches!(c, '*' | '~') {
            let run_len = run_length(&chars, i, c);
            if (1..=3).contains(&run_len) {
                let prev_word = i > 0 && is_word_char(chars[i - 1]);
                let next_word = i + run_len < chars.len() && is_word_char(chars[i + run_len]);
                if !(prev_word && next_word)
                    && let Some(close) =
                        find_run_at(&chars, &escaped_at, i + run_len + 1, c, run_len)
                {
                    let content: String = chars[i + run_len..close].iter().collect();
                    out.push_str(&strip_emphasis_and_links_at_depth(&content, depth + 1));
                    i = close + run_len;
                    continue;
                }
            }
            for _ in 0..run_len {
                out.push(c);
            }
            i += run_len;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn run_length(chars: &[char], from: usize, ch: char) -> usize {
    let mut n = 0;
    while from + n < chars.len() && chars[from + n] == ch {
        n += 1;
    }
    n
}

/// CommonMark's word character (the intraword-emphasis disqualifier).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Find a closing delimiter run: a window of exactly `len` unescaped `ch`
/// characters, itself a whole run — neither preceded nor followed by an
/// unescaped `ch` (an adjacent escaped one is literal text and does not
/// extend the run), and none of its characters backslash-escaped.
fn find_run_at(
    chars: &[char],
    escaped_at: &[bool],
    from: usize,
    ch: char,
    len: usize,
) -> Option<usize> {
    let mut i = from;
    while i + len <= chars.len() {
        let window = (0..len).all(|k| chars[i + k] == ch && !escaped_at[i + k]);
        let left_open = i == 0 || chars[i - 1] != ch || escaped_at[i - 1];
        let right_open = i + len == chars.len() || chars[i + len] != ch || escaped_at[i + len];
        if window && left_open && right_open {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Parse `<content>` starting at `open` (which points at `<`). The content
/// is the destination of an engine autolink — an absolute URI
/// (`scheme://…`) or a bare email address (html-to-markdown-rs's mailto
/// autolink emits the address without its scheme) — with no whitespace or
/// angle brackets inside. Returns the content and the index just past `>`.
fn parse_autolink(chars: &[char], open: usize) -> Option<(String, usize)> {
    let mut content = String::new();
    let mut i = open + 1;
    while i < chars.len() {
        match chars[i] {
            '>' => {
                if is_autolink_content(&content) {
                    return Some((content, i + 1));
                }
                return None;
            }
            c if c.is_whitespace() || c == '<' => return None,
            c => {
                content.push(c);
                i += 1;
            }
        }
    }
    None
}

fn is_autolink_content(content: &str) -> bool {
    if !content.is_ascii() || content.is_empty() {
        return false;
    }
    if let Some((scheme, rest)) = content.split_once("://") {
        !scheme.is_empty()
            && scheme
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
            && !rest.is_empty()
    } else {
        // a bare email: local@domain with a dot in the domain (the `@`
        // itself must pass, so the char sets are checked per part)
        match content.split_once('@') {
            Some((local, domain)) => {
                !local.is_empty()
                    && domain.contains('.')
                    && local.chars().all(|c| {
                        c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+' | '_' | '%')
                    })
                    && domain
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
            }
            None => false,
        }
    }
}

/// Parse `[label](url)` starting at `open` (which points at `[`). Returns
/// the label, the url, and the index just past the closing `)`. A `]`
/// inside the label arrives ESCAPED from anydoc (`\]`, its in-label
/// escaping), so bracket matching skips escaped ones.
fn parse_link(chars: &[char], escaped_at: &[bool], open: usize) -> Option<(String, String, usize)> {
    let close_bracket = find_unescaped_bracket_close(chars, escaped_at, open)?;
    let next = close_bracket + 1;
    if next >= chars.len() || chars[next] != '(' {
        return None;
    }
    let mut depth = 0;
    let mut i = next;
    while i < chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let label: String = chars[open + 1..close_bracket].iter().collect();
                    let url: String = chars[next + 1..i].iter().collect();
                    return Some((label, url, i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn find_unescaped_bracket_close(chars: &[char], escaped_at: &[bool], open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        if escaped_at[i] {
            continue;
        }
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Put the lifted code spans back, replacing each `\u{0}<idx>\u{0}`
/// sentinel with its span. The sentinel grammar is ours (NUL never
/// appears in engine markdown), but `strip` is a pub fn and literal NUL
/// reaches it directly: a sentinel-shaped sequence that is not one of
/// ours — a 0 (the indices are 1-based), an out-of-range index, a
/// non-numeric body, or an opener that never closes — is literal text
/// and passes through VERBATIM, NULs included. Never a panic (the old
/// `idx - 1` underflowed on 0: a debug abort, a release wrap to
/// usize::MAX), never a silent drop (an unmatched index used to vanish
/// with its NULs).
fn restore_code_spans(line: &str, spans: &[String]) -> String {
    if spans.is_empty() {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut index = String::new();
    let mut in_sentinel = false;
    for c in line.chars() {
        if c == '\u{0}' {
            if !in_sentinel {
                in_sentinel = true;
                index.clear();
            } else {
                if let Some(span) = index
                    .parse::<usize>()
                    .ok()
                    .and_then(|idx| idx.checked_sub(1))
                    .and_then(|idx| spans.get(idx))
                {
                    out.push_str(span);
                } else {
                    out.push('\u{0}');
                    out.push_str(&index);
                    out.push('\u{0}');
                }
                in_sentinel = false;
                index.clear();
            }
            continue;
        }
        if in_sentinel {
            index.push(c);
        } else {
            out.push(c);
        }
    }
    // An opener that never closes is literal text too: flush it verbatim.
    if in_sentinel {
        out.push('\u{0}');
        out.push_str(&index);
    }
    out
}

/// The strip contract, unit-pinned: every construct the four engines emit
/// (see the module docs' input contract) normalized to plain text, with the
/// entity policy as the one caller-chosen axis. The Python suite's
/// alignment gates test the same contract end-to-end over generated
/// documents; these are the per-construct pins.
#[cfg(all(test, feature = "documents"))]
mod tests {
    fn strip(markdown: &str) -> String {
        super::strip(markdown, false)
    }

    #[test]
    fn headings_keep_their_text_and_drop_the_markers() {
        assert_eq!(strip("# Title\n\n## Subtitle\n"), "Title\n\nSubtitle\n");
    }

    #[test]
    fn bullets_drop_markers_and_keep_indentation() {
        assert_eq!(
            strip("- first\n  - nested\n* also\n"),
            "first\n  nested\nalso\n"
        );
    }

    #[test]
    fn ordered_numbering_is_text_and_survives_the_paragraph_path() {
        // No engine's ordered marker is syntax the strip removes: `1. ` is
        // numbering text (anydoc emits `N. `, office_oxide `N. `, and the
        // strip's contract keeps ordered numbering), so it stays verbatim.
        assert_eq!(strip("1. first\n2. second\n"), "1. first\n2. second\n");
    }

    #[test]
    fn task_list_checkboxes_are_markers_not_content() {
        assert_eq!(strip("- [x] done\n- [ ] todo\n"), "done\ntodo\n");
    }

    #[test]
    fn blockquote_markers_go_but_the_text_stays() {
        assert_eq!(strip("> quoted line\n"), "quoted line\n");
    }

    #[test]
    fn nested_and_mixed_quote_list_heading_markers_all_strip() {
        // Probed 2026-09-09: anydoc renders each quote level with its own
        // `> ` (nested quotes are `> > text`), a list inside a quote as
        // `> - item`, a quote inside a list item as `- > text`, and a
        // heading inside a quote as `> ### heading`; html-to-markdown-rs
        // emits the same shapes (and `> > >` for a triple nest).
        assert_eq!(strip("> > inner quote\n"), "inner quote\n");
        assert_eq!(strip("> > > triple nested\n"), "triple nested\n");
        assert_eq!(strip("> - quoted item\n"), "quoted item\n");
        assert_eq!(strip("- > quoted in list\n"), "quoted in list\n");
        assert_eq!(strip("> ### quoted heading\n"), "quoted heading\n");
        // A blank quote line (`>` alone, anydoc's shape for an empty line
        // inside a quote) is a paragraph break, not text.
        assert_eq!(strip("> first\n>\n> second\n"), "first\n\nsecond\n");
    }

    #[test]
    fn quoted_table_rows_keep_cell_boundaries_and_drop_the_delimiter() {
        // Probed 2026-09-09 (html blockquote containing a table): the
        // engine emits `> | A | B |`; the quote marker goes, the row is a
        // table row — and the quoted `> | --- | --- |` delimiter row is
        // machinery, never text.
        let markdown = "> | A | B |\n> | --- | --- |\n> | 1 | 2 |\n";
        assert_eq!(strip(markdown), "A | B\n1 | 2\n");
    }

    #[test]
    fn marker_only_list_lines_carry_no_text() {
        // Probed 2026-09-09: html-to-markdown-rs emits a bare `-` line
        // for a list item whose only content is a blockquote, and empty
        // items arrive as marker-only lines. A marker with nothing after
        // it is not text.
        assert_eq!(strip("-\n> quoted in list\n"), "quoted in list\n");
        assert_eq!(strip("first\n-\nsecond\n"), "first\nsecond\n");
    }

    #[test]
    fn table_delimiter_rows_go_and_data_rows_keep_cell_boundaries() {
        let markdown = "| Unit | Status |\n| --- | --- |\n| T-101 | healthy |\n";
        assert_eq!(strip(markdown), "Unit | Status\nT-101 | healthy\n");
    }

    #[test]
    fn a_row_of_bare_dashes_is_data_not_a_delimiter() {
        // Measured 2026-09-09: a data row whose every cell is a bare dash
        // — a spreadsheet row of `-` placeholders — was dropped as a
        // separator row. The delimiter grammar takes a lone `-` only with
        // a colon's alignment intent; every engine's real delimiter is
        // `---` (or `:--`/`--:`/`:-:`).
        assert_eq!(strip("| - | - |\n"), "- | -\n");
        assert_eq!(strip("| a | -- | b |\n"), "a | -- | b\n");
        // the alignment-intent shapes stay machinery, engines' or GFM's
        assert_eq!(
            strip("| a | b |\n| :- | -: |\n| 1 | 2 |\n"),
            "a | b\n1 | 2\n"
        );
        assert_eq!(
            strip("| a | b |\n| :-: | --- |\n| 1 | 2 |\n"),
            "a | b\n1 | 2\n"
        );
        // one non-delimiter cell makes the whole row data
        assert_eq!(strip("| --- | x |\n"), "--- | x\n");
    }

    #[test]
    fn escaped_cell_pipes_stay_inside_their_cell() {
        // Both table-writing engines escape a literal pipe in a cell:
        // anydoc's TableCell escape arm and pdf_oxide's
        // `cell.text.replace('|', "\\|")`. Splitting on every `|` cut the
        // cell in half and stranded the backslash (measured 2026-09-09:
        // `| a \| b |` → `a \ | b`); the split honors unescaped pipes only.
        assert_eq!(strip("| a \\| b | plain |\n"), "a | b | plain\n");
        // An even backslash run before the pipe is a literal backslash
        // whose pipe IS the cell boundary (`\\|` = `\` + boundary).
        assert_eq!(strip("| x \\\\| y |\n"), "x \\ | y\n");
    }

    #[test]
    fn empty_header_rows_go_and_interior_empty_cells_stay_columns() {
        // anydoc renders an empty header row over a headerless table
        // (measured 2026-09-09 over an openpyxl sheet): all-empty rows are
        // machinery. Interior empty cells are column positions and stay.
        assert_eq!(strip("|  |  |\n| --- | --- |\n| a | b |\n"), "a | b\n");
        assert_eq!(strip("| a |  | b |\n"), "a |  | b\n");
    }

    #[test]
    fn table_cells_get_the_full_inline_pass() {
        // Cells are text: emphasis, code spans, links, and entities all
        // render (probed 2026-09-09: an html cell carrying
        // `<em>/<strong>/<code>/<a>` came out as raw markdown because the
        // cell path never ran the inline pass).
        assert_eq!(strip("| **b** *i* |\n"), "b i\n");
        assert_eq!(strip("| `c` |\n"), "c\n");
        assert_eq!(
            strip("| [l](https://e.example.com/z) |\n"),
            "l (https://e.example.com/z)\n"
        );
        assert_eq!(super::strip("| a &amp; b |\n", true), "a & b\n");
        // anydoc's cell break: multi-paragraph or hard-broken docx cells
        // render as `first<br>second`; a literal `<br>` in source text
        // arrives escaped (`\<br>`) and stays.
        assert_eq!(strip("| first<br>second |\n"), "first second\n");
        assert_eq!(strip("| keep \\<br> literal |\n"), "keep <br> literal\n");
    }

    #[test]
    fn code_fences_go_and_content_stays_verbatim() {
        let markdown = "```\nx = 1\n*y* stays literal\n```";
        assert_eq!(strip(markdown), "x = 1\n*y* stays literal\n");
    }

    #[test]
    fn blank_lines_inside_a_code_block_are_content_not_breaks() {
        // Measured 2026-09-09: html-to-markdown-rs (and anydoc's CodeBlock
        // renderer) keep interior blank lines verbatim — two blank lines
        // in a `<pre>` block stayed two in the markdown; the old
        // whole-string `\n\n\n` collapse ate one of them.
        let markdown = "```\na\n\n\nb\n```\n";
        assert_eq!(strip(markdown), "a\n\n\nb\n");
    }

    #[test]
    fn a_longer_fence_never_closes_on_an_interior_shorter_run() {
        // Probed 2026-09-09 (html `<pre>a\n```\nb</pre>`): the engine
        // emits a 4-backtick fence around code containing a ``` line —
        // both fence-writing engines pick the fence one longer than any
        // interior marker run. The old toggler flipped state on the
        // interior ``` (code after it was emphasis-stripped, the closer
        // re-opened a phantom fence, and post-closer text passed through
        // verbatim with its `*` markers intact). The tracker now closes
        // only on a run at least the OPENER's length.
        let markdown = "````\na\n```\nb\n````\n\nafter *emphasis* text\n";
        assert_eq!(strip(markdown), "a\n```\nb\n\nafter emphasis text\n");
        // a same-char run at the opener's exact length closes
        assert_eq!(strip("````\ncode\n````\n"), "code\n");
        // the length guard holds for tildes too
        assert_eq!(strip("~~~~\n~~~\nx\n~~~~\n"), "~~~\nx\n");
    }

    #[test]
    fn fences_close_on_their_own_marker_char_only() {
        // The old toggler flipped on EITHER marker char: a backtick fence
        // "closed" on a tilde run. Closers must match the opener's char.
        assert_eq!(strip("```\n~~~\n```\n"), "~~~\n");
        assert_eq!(strip("~~~\n```\n~~~\n"), "```\n");
    }

    #[test]
    fn a_line_starting_with_a_code_span_is_not_a_fence_opener() {
        // CommonMark's info-string rule: a backtick fence's info string
        // may not contain a backtick — so a line-opening ```x``` is an
        // inline code span (anydoc's serializer emits exactly that shape
        // for code containing backticks), never a fence. The old toggler
        // swallowed the rest of the document as phantom fence content.
        let markdown = "```x``` and then a *paragraph*\n";
        assert_eq!(strip(markdown), "x and then a paragraph\n");
        // a genuine opener's info string is backtick-free
        assert_eq!(strip("```rust\nlet x = 1;\n```\n"), "let x = 1;\n");
        // and a closer never carries an info string: this ```js line is
        // CONTENT of the open fence, not a close
        assert_eq!(
            strip("```\n```js\nstill code\n```\n"),
            "```js\nstill code\n"
        );
    }

    #[test]
    fn fenced_code_inside_a_blockquote_is_recognized_and_unmarked() {
        // Probed 2026-09-09 (html blockquote containing a `<pre>`; anydoc
        // renders the same shape — its BlockQuote renderer prefixes every
        // inner line with `> `): the fence opens and closes after the
        // quote marker, the content's `> ` machinery goes, the code stays
        // verbatim — emphasis inside it is NOT stripped, markers do not
        // leak. The old toggler never saw the fence (trim_start strips no
        // `>`): the ``` lines leaked as literal text and the code was
        // inline-stripped.
        let markdown = "> ```\n> quoted *code*\n> ```\n\nafter text\n";
        assert_eq!(strip(markdown), "quoted *code*\n\nafter text\n");
        // nested quotes close at their own depth: a singly-marked line is
        // content of the doubly-opened fence
        let nested = "> > ```\n> > deep *code*\n> > ```\n";
        assert_eq!(strip(nested), "deep *code*\n");
        // anydoc's empty quoted line (`>` alone) is an empty code line
        let blank = "> ```\n> a\n>\n> b\n> ```\n";
        assert_eq!(strip(blank), "a\n\nb\n");
        // a BARE fence run inside a quoted fence is content, not a close
        let stray = "> ```\n```\n> ```\n";
        assert_eq!(strip(stray), "```\n");
    }

    #[test]
    fn fenced_code_inside_a_list_item_is_recognized_and_unmarked() {
        // Probed 2026-09-09 (html list item containing a `<pre>`;
        // anydoc's render_list puts a first-block code fence right after
        // the marker and indents the closer by the marker width): the
        // fence opens on the bullet line, closes at the marker-width
        // indent, and the content's continuation indent is machinery.
        let markdown = "- ```\n  item *code*\n  ```\n";
        assert_eq!(strip(markdown), "item *code*\n");
        // a nested list's fence opens at indent 2 and closes at indent 4
        let nested = "- outer\n  * ```\n    deep *code*\n    ```\n";
        assert_eq!(strip(nested), "outer\ndeep *code*\n");
        // quote and list compose (html `<blockquote><ul><li><pre>…`):
        // `> - ``` ` opens, `>   ``` ` closes
        let quoted = "> - ```\n>   q list *code*\n>   ```\n";
        assert_eq!(strip(quoted), "q list *code*\n");
        // a BARE fence run while a bullet-context fence is open is
        // content, not a close
        let stray = "- ```\n```\n  ```\n";
        assert_eq!(strip(stray), "```\n");
    }

    #[test]
    fn tilde_fences_strip_like_backtick_ones() {
        // No tors-lane engine emits tildes by default (probed 2026-09-09:
        // html-to-markdown-rs's tilde style is an unset option;
        // anydoc/office_oxide/pdf_oxide write backticks or none) — the
        // accepted vocabulary is the strip's own contract, pinned here.
        assert_eq!(
            strip("~~~\nx = 1\n*y* stays literal\n~~~\n"),
            "x = 1\n*y* stays literal\n"
        );
        // info strings allowed (CommonMark: any content for tildes)
        assert_eq!(strip("~~~rust\nlet x = 1;\n~~~\n"), "let x = 1;\n");
        // in containers, same as backticks
        assert_eq!(strip("> ~~~\n> q ~code~\n> ~~~\n"), "q ~code~\n");
    }

    #[test]
    fn an_unterminated_fence_runs_to_the_end_as_content() {
        // The engine contract's degenerate tail: an opener with no closer
        // streams the rest of the document as code content (verbatim,
        // never inline-stripped) — the same doctrine fence_impl pins for
        // its own surface.
        assert_eq!(strip("```\nunterminated *code*\n"), "unterminated *code*\n");
    }

    #[test]
    fn fence_indent_trim_never_splits_a_multibyte_whitespace_char() {
        // Fuzz-found 2026-09-09 (cargo-fuzz gfm_strip, artifact
        // crash-bce7ac6e): a fence opened at indent 3 with a content line
        // whose leading whitespace run is two ASCII spaces plus the
        // two-byte NEXT LINE char \u{85} made the byte-budgeted machinery
        // trim slice at byte 3 — inside \u{85}'s bytes 2..4 — and panic
        // ("byte index 3 is not a char boundary"). The trim now consumes
        // only WHOLE leading whitespace chars while they fit the budget:
        // a char the budget cannot fit is the code's own content, kept
        // verbatim.
        let markdown = concat!("   ```\n", "  \u{85}x\n", "   ```\n");
        assert_eq!(strip(markdown), "\u{85}x\n");
    }

    #[test]
    fn horizontal_rules_go() {
        assert_eq!(strip("before\n\n---\n\nafter\n"), "before\n\nafter\n");
        // the dash rule's other spellings — interleaved whitespace, long
        // runs — are still rules
        assert_eq!(strip("before\n- - -\nafter\n"), "before\nafter\n");
        assert_eq!(strip("before\n----------\nafter\n"), "before\nafter\n");
    }

    #[test]
    fn star_and_underscore_runs_are_literal_text_not_rules() {
        // Probed 2026-09-09 (registry sources): every hr-emitting engine
        // writes `---` — html-to-markdown-rs's TagKind::Hr arm, anydoc's
        // Block::Rule, office_oxide's ThematicBreak; pdf_oxide emits no
        // rule at all. So `***`/`___` in engine markdown is literal text
        // from the unescaped engines (pdf-lane `_____` fill-in blanks and
        // `***` were dropped as rules), and anydoc escapes a literal
        // `_`/`*` so a bare run cannot be its output either.
        assert_eq!(strip("name\n_____\nvalue\n"), "name\n_____\nvalue\n");
        assert_eq!(strip("blank line\n___\nhere\n"), "blank line\n___\nhere\n");
        assert_eq!(strip("stars\n***\nafter\n"), "stars\n***\nafter\n");
    }

    #[test]
    fn links_become_label_and_url_images_become_alt_text() {
        assert_eq!(
            strip("See the [handbook](https://example.com/x) and ![logo](img.png).\n"),
            "See the handbook (https://example.com/x) and logo.\n"
        );
    }

    #[test]
    fn links_whose_label_is_the_url_collapse_to_the_label() {
        // pdf_oxide renders a bare URL as `[url](url)` (and a bare email
        // as `[e](mailto:e)`); html-to-markdown-rs emits `<url>` for the
        // same link. Same link, two syntaxes — one text shape, the
        // destination once.
        assert_eq!(
            strip("go to [https://e.example.com/x](https://e.example.com/x) now\n"),
            "go to https://e.example.com/x now\n"
        );
        assert_eq!(
            strip("mail [u@e.com](mailto:u@e.com) now\n"),
            "mail u@e.com now\n"
        );
    }

    #[test]
    fn autolinks_render_the_destination_without_angle_brackets() {
        // html-to-markdown-rs, `autolinks: true` (its default): an <a>
        // whose text is its absolute-URI href, or a mailto whose text is
        // the bare address, renders as `<scheme://…>` / `<user@…>`
        // (probed 2026-09-09). Other angle-bracket content is not an
        // autolink and passes through.
        assert_eq!(
            strip("see <https://example.com/x> there\n"),
            "see https://example.com/x there\n"
        );
        assert_eq!(
            strip("mail <user@example.com> now\n"),
            "mail user@example.com now\n"
        );
        assert_eq!(
            strip("keep <not a link> here\n"),
            "keep <not a link> here\n"
        );
        assert_eq!(strip("math <x> y\n"), "math <x> y\n");
    }

    #[test]
    fn emphasis_markers_unwrap() {
        // `*`/`**`/`***`/`~~` are the engine vocabulary (anydoc and
        // office_oxide style runs, html-to-markdown-rs's `<em>`/`<strong>`
        // /`<del>`, pdf_oxide's bold detection).
        assert_eq!(
            strip("**bold** and *italic* and ***both*** and ~~gone~~\n"),
            "bold and italic and both and gone\n"
        );
    }

    #[test]
    fn underscores_are_literal_text_never_emphasis() {
        // No engine emits `_` as an emphasis marker (measured 2026-09-09
        // across all four: anydoc/office_oxide/html style with `*`,
        // pdf_oxide's bold detection emits `**`), and anydoc — the one
        // engine that escapes markdown-significant text — deliberately
        // leaves `_` BARE where CommonMark cannot pair it (intraword or
        // whitespace-flanked). Bare `_` is therefore always literal text:
        // snake_case stays, spaced singles stay, and a pairable
        // `_emphasis_-shaped` run in unescaped engines' literal text stays
        // too — CommonMark's word-start/word-end rule would strip
        // ` _under_ ` from a PDF's literal text (pdf_oxide escapes
        // nothing), which is corruption, not normalization. Escaped
        // `\_` un-escapes to the same literal.
        assert_eq!(
            strip("foo_bar_baz and a1_b2_c3\n"),
            "foo_bar_baz and a1_b2_c3\n"
        );
        // The measured anydoc corruption this pins: `a _ b and _ c`
        // (both underscores whitespace-flanked, emitted bare precisely
        // because they cannot pair) used to become `a  b and  c`.
        assert_eq!(strip("a _ b and _ c spaced\n"), "a _ b and _ c spaced\n");
        assert_eq!(strip("spaced _under_ words\n"), "spaced _under_ words\n");
        assert_eq!(
            super::strip("escaped \\_under\\_ marks\n", true),
            "escaped _under_ marks\n"
        );
    }

    #[test]
    fn intraword_star_and_tilde_runs_stay_whole() {
        // pdf_oxide, html-to-markdown-rs, and office_oxide emit literal
        // text unescaped: `3*4*5` in a PDF is arithmetic, so an intraword
        // run is literal (word characters on both flanks disqualify it).
        // And a run that does not pair passes through WHOLE — re-reading
        // the tail of a disqualified `**` as a fresh `*` run used to
        // half-strip it (measured 2026-09-09: anydoc's and office_oxide's
        // mid-word bold `foo**bar**baz` came out `foo*bar*baz`).
        assert_eq!(strip("3*4*5 and 2**3**4\n"), "3*4*5 and 2**3**4\n");
        assert_eq!(strip("foo**bar**baz\n"), "foo**bar**baz\n");
        assert_eq!(strip("H~2~O formula\n"), "H~2~O formula\n");
        // The measured, accepted price: an intraword `<em>` (html) or
        // mid-word bold run (anydoc/office_oxide) leaves its markers in
        // the text — indistinguishable in markdown from PDF literal text.
    }

    #[test]
    fn inline_code_keeps_content_without_backticks() {
        assert_eq!(strip("run `cargo test` now\n"), "run cargo test now\n");
    }

    #[test]
    fn backslash_escaped_backticks_are_literal_text_not_a_code_span() {
        // anydoc's exact shape for a literal backtick in source text: the
        // opening backtick escaped, the closing one bare. Both are literal
        // text; the strip must keep them, not pair them into a code span
        // and strand the backslash (the regression this pins).
        assert_eq!(
            super::strip("backticks \\`inline` shaped text", false),
            "backticks `inline` shaped text\n"
        );
        // and a GENUINE code span still lifts (escapes untouched inside):
        assert_eq!(strip("real `code span` here\n"), "real code span here\n");
        // an escaped pair — both backticks escaped — is literal too
        assert_eq!(
            super::strip("both \\`open\\` and close", false),
            "both `open` and close\n"
        );
    }

    #[test]
    fn code_spans_are_lifted_before_marker_stripping() {
        // The `*` inside a code span is literal text, not emphasis: the
        // span is restored verbatim, markers intact.
        assert_eq!(strip("use `a * b` here\n"), "use a * b here\n");
    }

    #[test]
    fn code_spans_close_on_equal_length_backtick_runs() {
        // anydoc's serializer (its `backtick_fence`) emits a fence one
        // longer than any backtick run in the code, plus CommonMark's
        // one-space pad when the content starts or ends with a backtick —
        // measured 2026-09-09 over an epub `<code>` source:
        //   run ``a`b`` now      → the interior single backtick is content
        //   lead `` `tick `` trail  → the pad spaces are fence machinery
        //   double ```x``y``` run   → a double run is content for a triple fence
        assert_eq!(strip("run ``a`b`` now\n"), "run a`b now\n");
        assert_eq!(strip("lead `` `tick `` trail\n"), "lead `tick trail\n");
        assert_eq!(strip("double ```x``y``` run\n"), "double x``y run\n");
        // An unpaired run stays literal, whole.
        assert_eq!(strip("just `open\n"), "just `open\n");
        assert_eq!(strip("two `` open\n"), "two `` open\n");
        // A run whose only later partner is a different length does not
        // pair with it.
        assert_eq!(strip("a `b`` c\n"), "a `b`` c\n");
    }

    #[test]
    fn hardbreak_markers_go_but_literal_backslashes_stay() {
        // anydoc ends an intra-paragraph hard break with a bare `\` (odd
        // trailing backslash run; its literals are always `\\`), so on the
        // anydoc lane an odd trailing run loses exactly one backslash.
        // html-to-markdown-rs and office_oxide mark hard breaks with
        // trailing SPACES — already dropped by line trim (pinned below) —
        // and pdf_oxide emits no markers, so the drop is anydoc-lane only
        // and a PDF's literal `C:\` survives untouched.
        assert_eq!(
            super::strip("before break\\\nafter break\n", true),
            "before break\nafter break\n"
        );
        // an even run is a literal backslash (escaped `\\` → one `\`)
        assert_eq!(super::strip("trailing \\\\\n", true), "trailing \\\n");
        // literal pair + marker (three raw backslashes) keeps the literal
        assert_eq!(super::strip("end \\\\\\\nnext", true), "end \\\nnext\n");
        // off the anydoc lane the marker is never dropped
        assert_eq!(super::strip("path C:\\\n", false), "path C:\\\n");
    }

    #[test]
    fn space_style_hardbreaks_lose_only_their_spaces() {
        // html-to-markdown-rs (NewlineStyle::Spaces, its default) and
        // office_oxide mark a hard break with two trailing spaces; line
        // trimming already removes them, and the break stays a break.
        assert_eq!(strip("line one  \nline two\n"), "line one\nline two\n");
    }

    #[test]
    fn structure_lines_get_the_same_inline_pass_as_paragraphs() {
        // Measured 2026-09-09: a docx heading or list item carrying a
        // hyperlink (and a heading carrying an entity) came out of to_text
        // as raw `[label](url)` / `&amp;` because the structure arms
        // returned their text as-is. Structure strips markers; TEXT is
        // still text.
        assert_eq!(
            strip("# See the [handbook](https://h.example.com/t) now\n"),
            "See the handbook (https://h.example.com/t) now\n"
        );
        assert_eq!(
            strip("- [bullet link](https://h.example.com/b)\n"),
            "bullet link (https://h.example.com/b)\n"
        );
        assert_eq!(strip("- **bold** item\n"), "bold item\n");
        assert_eq!(
            super::strip("# Torque &amp;amp; figures\n", true),
            "Torque &amp; figures\n"
        );
    }

    #[test]
    fn unbalanced_markup_passes_through_unchanged() {
        assert_eq!(
            strip("a [ bare bracket and * lone mark\n"),
            "a [ bare bracket and * lone mark\n"
        );
    }

    #[test]
    fn entity_policy_is_the_caller_chosen_axis() {
        let markdown = "Torque &amp; figures\n";
        assert_eq!(super::strip(markdown, false), "Torque &amp; figures\n");
        assert_eq!(super::strip(markdown, true), "Torque & figures\n");
    }

    #[test]
    fn paragraph_breaks_survive_and_trailing_newlines_collapse_to_one() {
        assert_eq!(strip("one\n\ntwo\n\n\n"), "one\n\ntwo\n");
        // runs of blanks collapse to one break by construction (the
        // pending-break counter), including around dropped lines
        assert_eq!(strip("one\n\n\n\n\ntwo\n"), "one\n\ntwo\n");
        assert_eq!(strip(""), "");
        assert_eq!(strip("\n"), "");
    }

    #[test]
    fn pathological_link_label_nesting_degrades_instead_of_overflowing() {
        // RED-GREEN 2026-09-09: `strip_emphasis_and_links` recursed once
        // per bracket-nesting level of a link label with no bound at all.
        // Measured: a docx whose paragraph text is `[[[…]]()…]()` ~30,000
        // deep (~120 KB) made `to_text(backend="oxide")` SIGSEGV (exit
        // -11) — the recursion runs on the caller's stack, and no binding
        // in any language can catch a stack overflow. This drives the
        // public `strip` entry with a 200,000-deep construct: deep enough
        // to exhaust every stack this crate runs on. The contract past
        // the cap (MAX_INLINE_DEPTH, 256): the remaining label is emitted
        // as literal text — lossy (its own `[…](…)` machinery stays) but
        // alive, the same degradation the measured engines show on deep
        // HTML.
        let depth = 200_000;
        let markdown = format!("{}x{}\n", "[".repeat(depth), "]()".repeat(depth));
        let text = strip(&markdown);
        // The degradation is exact: each level below the cap unwraps one
        // `[…](…)` layer, and the first call AT the cap returns its whole
        // line verbatim — 256 wrapper levels gone, the rest literal text.
        let remaining = depth - 256;
        let expected = format!("{}x{}\n", "[".repeat(remaining), "]()".repeat(remaining));
        assert_eq!(text, expected);
    }

    #[test]
    fn literal_nul_sentinel_sequences_are_text_not_a_crash_or_a_drop() {
        // RED-GREEN 2026-09-09: the engines strip NUL from their text, so
        // the `\u{0}<idx>\u{0}` sentinels [`lift_code_spans`] writes are
        // unambiguous in engine markdown — but `strip` is a pub fn (and a
        // fuzz target), and literal NUL reaches it directly. A `\u{0}0\u{0}`
        // body (0 is one BELOW the first sentinel, which is 1-based) hit
        // `spans.get(idx - 1)`: debug panicked on the underflow; release
        // wrapped to usize::MAX, `.get()` → `None`, and the text between
        // the NULs silently vanished. The contract pinned here: a
        // sentinel-shaped sequence that is not one of ours — a 0, an
        // out-of-range index, a non-numeric body, or an opener that never
        // closes — is literal text and passes through verbatim, NULs
        // included. Never a panic, never a silent drop. (The real code
        // span in the input keeps `spans` non-empty: an empty-span line
        // short-circuits restore and exercises nothing.)
        let input = "real `code` and \u{0}0\u{0} \u{0}9\u{0} \u{0}x\u{0} \u{0} tail\n";
        assert_eq!(
            strip(input),
            "real code and \u{0}0\u{0} \u{0}9\u{0} \u{0}x\u{0} \u{0} tail\n"
        );
    }

    // --- probed and NOT emitted: the shapes the strip deliberately does
    // not handle, with the measurement that closed each question
    // (2026-09-09). House doctrine: the tests module documents what was
    // measured.
    //
    // SETEXT HEADINGS (`text\n====`): not emitted by any engine — anydoc
    // renders ATX only (`"#".repeat(level)`), html-to-markdown-rs defaults
    // to HeadingStyle::Atx, pdf_oxide detects headings as `#`/`##`/`###`,
    // office_oxide emits `#`. Nothing to strip; a literal `====` line from
    // unescaped engines is text and stays.
    //
    // `\r\n` LINE ENDINGS: the strip splits on `\n` and `trim_end`s each
    // line, so a stray `\r` never survives outside fences; no engine
    // emits `\r` at all (probed: a CRLF-written HTML file converts
    // byte-identically to the LF-written one, `<pre>` included — the
    // engine normalizes; anydoc/office_oxide/pdf_oxide build lines with
    // `\n` directly). No fence-interior `\r` trimming: unprobed shape.
    //
    // MULTI-LINE INLINE CODE SPANS: no engine emits a code span split
    // across lines (anydoc replaces a span's newline with a space; the
    // html engine's in-code `<br>` shape was not emitted by our corpus
    // generators). Per-line lifting leaves both halves as literal
    // backticks if one ever appears — re-probe before handling.
}
