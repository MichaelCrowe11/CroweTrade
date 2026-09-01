// Test-runner shim: the engine imports sibling modules with `.js` extensions
// (what the Worker bundler wants); node --test with strip-types wants the
// `.ts` file that actually exists. Map one to the other, nothing else.
import { register } from "node:module"
import { pathToFileURL } from "node:url"
register(new URL("./ts-hooks.mjs", import.meta.url), pathToFileURL("./"))
