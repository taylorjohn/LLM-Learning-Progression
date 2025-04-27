# Assignment 1 Writeup

## Problem (unicode): Understanding Unicode (Page 3)

**(a) What Unicode character does `chr(0)` return?**

`chr(0)` returns the **Null character** (often represented as `\0` or `NUL`). It's a control character with the code point 0, typically used as a terminator or placeholder in various contexts (e.g., null-terminated strings in C), although it doesn't usually have a visible representation.

**(b) How does this character's string representation (`__repr__`) differ from its printed representation?**

*   **Printed Representation (`print(chr(0))`):** When printed directly, the Null character usually produces no visible output or might cause the terminal to behave unexpectedly, as it's a non-printing control character.
*   **String Representation (`repr(chr(0))`):** The `repr()` function aims to provide an unambiguous, developer-friendly representation of the object. For `chr(0)`, it typically returns the string `'\x00'`. This explicitly shows the hexadecimal value of the byte representing the Null character, making it clear which character it is, unlike the (often invisible) printed output.

**(c) What happens when this character occurs in text?**

When the Null character (`\x00`) appears within a text string in many programming contexts (especially those influenced by C, like Python file I/O or string processing in some libraries):

*   **String Termination:** It can be interpreted as the end of the string, causing the rest of the text after the Null character to be ignored or truncated.
*   **File Handling:** Reading or writing text files containing null bytes can sometimes lead to errors or unexpected behavior, as some text processing functions are not designed to handle them gracefully, especially if expecting standard text encodings like UTF-8 where null bytes are not typically part of valid character representations (except as terminators).
*   **Rendering Issues:** Displaying text containing null characters in UIs or terminals might lead to rendering glitches or unexpected formatting.

## Problem (unicode2): Unicode Encodings (Page 4)

**(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?**

1.  **Compatibility & Ubiquity:** UTF-8 is the dominant character encoding on the web and in most modern systems. Training on UTF-8 ensures maximum compatibility with real-world text data without needing transcoding.
2.  **Space Efficiency for ASCII:** UTF-8 uses only one byte for ASCII characters (0-127), which are very common in English text and code. UTF-16 uses two bytes, and UTF-32 uses four bytes for these characters, leading to significantly larger data sizes for ASCII-heavy text.
3.  **Variable-Length Advantage:** While variable-length encoding adds complexity, it allows UTF-8 to represent the entire Unicode range efficiently. It avoids the fixed 2-byte (UTF-16, sometimes needing surrogate pairs for characters beyond the Basic Multilingual Plane) or 4-byte (UTF-32) representations, which can be wasteful for common characters.
4.  **No Byte Order Mark (BOM) Issues:** UTF-8 does not strictly require a BOM, simplifying processing compared to UTF-16 where byte order matters and BOMs are often used.
5.  **Robustness:** UTF-8 is designed such that sequence errors are often localized, making it somewhat more robust to corruption compared to multi-byte fixed-width encodings.

**(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why does this function yield incorrect results? Provide an example input byte string for which `decode_utf8_bytes_to_str_wrong` produces the wrong result.**

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([chr(b) for b in bytestring])
```

**Why it's wrong:** This function incorrectly assumes a one-to-one mapping between bytes and Unicode code points. It treats each byte in the input `bytestring` as an independent Unicode code point by calling `chr()` on each byte's integer value. However, UTF-8 is a *variable-length* encoding. Code points beyond the basic ASCII range (127) are represented using *multiple* bytes (2, 3, or 4 bytes). This function splits multi-byte sequences apart, interpreting each byte individually as a separate (and usually incorrect) character.

**Example:**

Let's take the Japanese character "こ" (ko).

*   Its Unicode code point is U+3053.
*   Its UTF-8 encoding is the 3-byte sequence: `0xE3 0x81 0x93`.

```python
correct_bytes = "こ".encode('utf-8') # -> b'\xe3\x81\x93'

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([chr(b) for b in bytestring])

# Correct decoding:
print(f"Correct decoding: {correct_bytes.decode('utf-8')}")

# Incorrect decoding:
incorrect_result = decode_utf8_bytes_to_str_wrong(correct_bytes)
print(f"Incorrect function output: {incorrect_result}")
print(f"Incorrect repr: {repr(incorrect_result)}")
# The individual bytes correspond to these code points:
# hex E3 -> dec 227 -> chr(227) -> 'ã' (Latin Small Letter A with Tilde)
# hex 81 -> dec 129 -> chr(129) -> Undefined/Control Character (often renders oddly or not at all)
# hex 93 -> dec 147 -> chr(147) -> 'ô' (Latin Small Letter O with Circumflex) in some legacy encodings, or Undefined.
# Expected output might be something like 'ã\x81\x93' or similar depending on system
```

The incorrect function breaks the 3-byte sequence `b'\xe3\x81\x93'` into three separate bytes and calls `chr()` on each (227, 129, 147). This results in a string of three unrelated characters (e.g., `'ã\x81\x93'` or similar, depending on how control characters are represented), completely different from the intended single character "こ".

**(c) Give a two byte sequence that does not correspond to any Unicode character.**

Many two-byte sequences are invalid in UTF-8. An invalid sequence occurs if the bytes don't follow the specific UTF-8 pattern (start bits indicating sequence length, continuation bits being `10xxxxxx`).

**Example:** `b'\xC2\x20'`

*   `0xC2` (binary `11000010`) indicates the start of a 2-byte sequence.
*   The *next* byte *must* be a continuation byte, starting with `10` (binary).
*   `0x20` (binary `00100000`) is the ASCII space character. It does *not* start with `10`.

Therefore, `b'\xC2\x20'` is an invalid UTF-8 sequence. Attempting to decode it using standard UTF-8 decoders will raise an error:

```python
invalid_sequence = b'\xc2\x20'
try:
    invalid_sequence.decode('utf-8')
except UnicodeDecodeError as e:
    print(f"Decoding failed as expected: {e}")
# Output: Decoding failed as expected: 'utf-8' codec can't decode byte 0x20 in position 1: invalid start byte
```

Another simple example is any byte that is defined as a continuation byte (`10xxxxxx`) appearing where a start byte is expected, e.g., `b'\x80'`. A lone continuation byte is invalid. 