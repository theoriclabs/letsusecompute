(function () {
  const re =
    /(#.*$)|("[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*')|((?<=\s)(?:--[\w-]+|-[A-Za-z]+)\b)|((?<=^|\s|\|)(?:compute|git|curl|cd|sh|export|cat|echo)\b)/gm;

  function escape(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function highlight(src) {
    let out = "";
    let cursor = 0;
    for (const match of src.matchAll(re)) {
      const index = match.index ?? 0;
      out += escape(src.slice(cursor, index));
      if (match[1]) out += `<span class="token comment">${escape(match[1])}</span>`;
      else if (match[2]) out += `<span class="token string">${escape(match[2])}</span>`;
      else if (match[3]) out += `<span class="token attr-name">${escape(match[3])}</span>`;
      else out += `<span class="token function">${escape(match[4])}</span>`;
      cursor = index + match[0].length;
    }
    return out + escape(src.slice(cursor));
  }

  document.querySelectorAll("pre code.language-bash").forEach((block) => {
    block.innerHTML = highlight(block.textContent);
  });
})();
