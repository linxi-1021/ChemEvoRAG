import type { MarkdownBlock } from '../types';

/**
 * Parse full.md content into logical blocks for the Markdown tab.
 * Splits at headings, images, tables, details, and blank lines.
 */
export function parseMarkdownToBlocks(md: string): MarkdownBlock[] {
  const lines = md.split('\n');
  const blocks: MarkdownBlock[] = [];
  let blockIdx = 0;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Heading
    const headingMatch = line.match(/^(#{1,6})\s+(.*)/);
    if (headingMatch) {
      blocks.push({
        id: `md_${blockIdx++}`,
        type: 'heading',
        content: line,
        level: headingMatch[1].length,
      });
      i++;
      continue;
    }

    // Image
    const imgMatch = line.match(/^!\[.*?\]\((images\/[a-f0-9]+\.jpg)\)/);
    if (imgMatch) {
      const imageId = imgMatch[1].replace('images/', '').replace('.jpg', '');
      blocks.push({
        id: `md_${blockIdx++}`,
        type: 'image',
        content: line,
        imageUrl: imgMatch[1],
        imageId,
      });
      i++;

      // Collect following <details> block if present
      if (i < lines.length && lines[i].trim() === '<details>') {
        let detailsContent = lines[i];
        i++;
        while (i < lines.length && !lines[i].trim().includes('</details>')) {
          detailsContent += '\n' + lines[i];
          i++;
        }
        if (i < lines.length) {
          detailsContent += '\n' + lines[i];
          i++;
        }
        blocks.push({
          id: `md_${blockIdx++}`,
          type: 'chemical',
          content: detailsContent,
        });
      }
      continue;
    }

    // HTML table
    if (line.trim().startsWith('<table>')) {
      let tableContent = line;
      while (i < lines.length && !lines[i].trim().includes('</table>')) {
        i++;
        tableContent += '\n' + lines[i];
      }
      i++;
      blocks.push({
        id: `md_${blockIdx++}`,
        type: 'table',
        content: tableContent,
      });
      continue;
    }

    // Details block (chemical annotation)
    if (line.trim().startsWith('<details>')) {
      let detailsContent = line;
      while (i < lines.length && !lines[i].trim().includes('</details>')) {
        i++;
        detailsContent += '\n' + lines[i];
      }
      if (i < lines.length) {
        detailsContent += '\n' + lines[i];
        i++;
      }
      blocks.push({
        id: `md_${blockIdx++}`,
        type: 'chemical',
        content: detailsContent,
      });
      continue;
    }

    // Footnote lines [a], [b], etc.
    if (/^\[.?\]\s/.test(line.trim())) {
      blocks.push({
        id: `md_${blockIdx++}`,
        type: 'footnote',
        content: line,
      });
      i++;
      continue;
    }

    // Empty line — skip
    if (line.trim() === '') {
      i++;
      continue;
    }

    // Regular paragraph — accumulate consecutive non-empty lines
    let paraContent = line;
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].match(/^#{1,6}\s/) &&
      !lines[i].match(/^!\[.*?\]\(/) &&
      !lines[i].trim().startsWith('<table>') &&
      !lines[i].trim().startsWith('<details>') &&
      !/^\[.?\]\s/.test(lines[i].trim())
    ) {
      paraContent += '\n' + lines[i];
      i++;
    }
    blocks.push({
      id: `md_${blockIdx++}`,
      type: 'paragraph',
      content: paraContent,
    });
  }

  return blocks;
}
