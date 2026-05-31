/**
 * Coordinate conversion utilities for bbox overlay on PDF pages.
 * Transforms MinerU internal coordinates to pdf.js viewport coordinates.
 */

/** Transform a bbox from MinerU canvas coords to pdf.js viewport coords */
export function transformBbox(
  bbox: [number, number, number, number],
  mineruWidth: number,
  mineruHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): [number, number, number, number] {
  const scaleX = viewportWidth / mineruWidth;
  const scaleY = viewportHeight / mineruHeight;
  return [
    bbox[0] * scaleX,
    bbox[1] * scaleY,
    bbox[2] * scaleX,
    bbox[3] * scaleY,
  ];
}
