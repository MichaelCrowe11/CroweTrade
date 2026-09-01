import { existsSync } from "node:fs"
import { fileURLToPath } from "node:url"
export async function resolve(specifier, context, next) {
  if (specifier.endsWith(".js") && (specifier.startsWith("./") || specifier.startsWith("../")) && context.parentURL) {
    const abs = new URL(specifier, context.parentURL)
    if (!existsSync(fileURLToPath(abs))) return next(specifier.slice(0, -3) + ".ts", context)
  }
  return next(specifier, context)
}
