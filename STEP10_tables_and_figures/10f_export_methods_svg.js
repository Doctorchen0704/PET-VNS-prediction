/* Export an author-approved, editable methods SVG. No analysis or redesign. */
const fs = require('fs');
const path = require('path');
async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.length < 2) {
    console.log('node 10f_export_methods_svg.js INPUT.svg NEW_OUTPUT_DIRECTORY [--dpi 150|600]');
    console.log('Requires the ordinary npm package sharp. Keeps the approved SVG composition.');
    return;
  }
  const input = path.resolve(args[0]), out = path.resolve(args[1]);
  const ix = args.indexOf('--dpi'), dpi = ix < 0 ? 150 : Number(args[ix+1]);
  if (!Number.isInteger(dpi) || dpi < 72 || dpi > 1200) throw new Error('Explicit sensible DPI required');
  if (!fs.existsSync(input) || !input.toLowerCase().endsWith('.svg')) throw new Error('Existing approved SVG required');
  if (fs.existsSync(out)) throw new Error('Use a fresh output directory');
  const sharp = require('sharp');
  const text = fs.readFileSync(input, 'utf8');
  const view = text.match(/viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"/);
  if (!view) throw new Error('SVG viewBox required to preserve physical size');
  const width = Math.round(Number(view[3]) / 72 * dpi), height = Math.round(Number(view[4]) / 72 * dpi);
  // Set a pixel viewport without altering the vector geometry or image elements.
  const viewport = text.replace(/width="[^"]+" height="[^"]+"/, `width="${width}px" height="${height}px"`);
  fs.mkdirSync(out, {recursive:true});
  fs.copyFileSync(input, path.join(out, 'Figure1.svg'));
  const rgb = await sharp(Buffer.from(viewport), {density:72,limitInputPixels:false})
    .flatten({background:'#ffffff'}).removeAlpha().toColourspace('srgb').raw().toBuffer();
  const image = sharp(rgb, {raw:{width,height,channels:3}});
  await image.clone().png().withMetadata({density:dpi}).toFile(path.join(out,'Figure1.png'));
  await image.clone().tiff({compression:'lzw'}).withMetadata({density:dpi}).toFile(path.join(out,'Figure1.tiff'));
  fs.writeFileSync(path.join(out,'export.json'), JSON.stringify({dpi,width,height,
    source:'Caller-supplied approved editable SVG', composition_changed:false,
    analyses_run:false, visual_check:'REQUIRED', sharp:sharp.versions},null,2)+'\n');
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});
