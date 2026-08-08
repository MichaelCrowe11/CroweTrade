// Domain logic lives in shared/ so the terminal and the cloud engine run the
// SAME policy code — a verdict rendered on screen and a verdict that opens a
// paper position must never be able to disagree. This file is a pass-through.
export * from "../../../shared/gates.js"
