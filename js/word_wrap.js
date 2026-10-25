function cloneAttributes(element, sourceNode) {
  let attr;
  let attributes = Array.prototype.slice.call(sourceNode.attributes);
  while(attr = attributes.pop()) {
    element.setAttribute(attr.nodeName, attr.nodeValue);
  }
}

window.addEventListener('DOMContentLoaded', () => {
	let re = /^[ \t]+/;
	document.querySelectorAll(".main-content pre code").forEach((code) => {
		// Zola's highlighter puts every line in its own `.giallo-l` span with the tokens inside
		let lines = code.querySelectorAll(":scope > .giallo-l");
		if (lines.length == 0) {
			return;
		}
		let new_code = document.createElement("code");
		new_code.classList = code.classList;
		cloneAttributes(new_code, code);
		lines.forEach((line) => {
			let table = document.createElement("span");
			let tr = document.createElement("span");
			let td1 = document.createElement("span");
			let td2 = document.createElement("span");
			let span1 = document.createElement("span");
			let span1_hidden = document.createElement("span");

			table.classList.add("table");
			tr.classList.add("tr");
			td1.classList.add("td");
			td2.classList.add("td");

			td1.classList.add("no-copy");
			span1_hidden.classList.add("hidden-copy");

			let tokens = Array.from(line.children);
			let first = tokens.length > 0 ? tokens[0] : document.createElement("span");
			let string = first.innerHTML;
			let matched = string.match(re);
			if (matched == null) {
				matched = [""];
			}

			let span2 = first.cloneNode();
			span1.innerHTML = matched[0];
			span1_hidden.innerHTML = matched[0];
			span2.innerHTML = string.slice(matched[0].length);

			td1.appendChild(span1);
			td2.appendChild(span1_hidden);
			td2.appendChild(span2);
			tokens.slice(1).forEach((span) => {
				td2.appendChild(span.cloneNode(true));
			});
			let line_end = document.createElement("span");
			line_end.innerHTML = "\n";
			td2.appendChild(line_end);

			tr.appendChild(td1);
			tr.appendChild(td2);
			table.appendChild(tr);
			new_code.appendChild(table);
		});
		code.parentNode.replaceChild(new_code, code);
	});
});
