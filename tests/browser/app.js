(function () {
  "use strict";

  function safeId(value) {
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "-");
  }

  function render(payload) {
    var answer = document.getElementById("answer");
    answer.textContent = payload.response || payload.text || "";
    answer.setAttribute("dir", payload.direction || "ltr");
    if (payload.language) {
      answer.setAttribute("lang", payload.language);
    }

    var probe = document.getElementById("bidi-probe");
    probe.textContent = answer.textContent;
    probe.setAttribute("dir", payload.direction || "ltr");

    var citations = document.getElementById("citations");
    citations.textContent = "";
    (payload.citations || []).forEach(function (citation) {
      var figure = document.createElement("figure");
      figure.className = "citation-block";
      figure.id = "citation-block-" + safeId(citation.reference);

      var quote = document.createElement("blockquote");
      quote.className = "citation-quote";
      quote.textContent = citation.quote || "";
      quote.setAttribute("dir", citation.quote_direction || "rtl");
      figure.appendChild(quote);

      var link = document.createElement("a");
      link.className = "citation-link";
      link.id = "citation-" + safeId(citation.reference);
      link.href = "#" + encodeURIComponent(citation.reference);
      link.textContent = citation.reference;
      link.setAttribute("data-reference", citation.reference);
      link.setAttribute("aria-label", "Quran citation " + citation.reference);
      figure.appendChild(link);

      citations.appendChild(figure);
    });

    var button = document.getElementById("copy-button");
    var output = document.getElementById("copy-payload");
    button.addEventListener("click", function () {
      var copied = answer.textContent || "";
      output.textContent = copied;
      output.setAttribute("data-copied-text", copied);
      window.__copiedText = copied;
    });

    window.__fixture = payload;
    window.__ready = true;
  }

  var params = new URLSearchParams(window.location.search);
  var name = params.get("fixture") || "arabic.json";
  fetch("fixtures/" + encodeURIComponent(name), { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) {
        throw new Error("fixture load failed: " + response.status);
      }
      return response.json();
    })
    .then(render)
    .catch(function (error) {
      document.getElementById("answer").textContent = "Harness error: " + error.message;
      window.__ready = true;
      window.__error = error.message;
    });
})();
