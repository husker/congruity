# Data and execution boundaries

Congruity is a local research tool. Treat model output, downloaded papers and
generated candidate code as untrusted inputs.

## Where data goes

- Research queries go to OpenAlex. Composition prompts and selected paper content
  go to the model provider used by your authenticated Codex CLI. Optional relevance
  providers also receive the material they evaluate.
- Run artifacts can include complete prompts, model output, stderr, local paths
  and source excerpts. They are private working data by default. The generated
  dashboard can expose their content too.
- `.env` is loaded locally. Use `.env.example` for configuration names, not real
  credentials. Do not put confidential research or client data in a run unless
  its use with the configured services has been authorized.
- The optional header-fetching experiment saves only allowlisted protocol
  metadata. Cookies, authentication headers and vendor session identifiers are
  excluded. Keep captures local anyway; they are not publication fixtures.

## Executing candidates

`autoverify` and `autosearch` run generated Python through local benchmark
processes. The harness checks results and applies timeouts, but it does not
isolate the process from your machine, environment, files or network. Instructions
in a prompt are not an access-control mechanism.

Run candidate experiments in an isolated environment with no unrelated secrets
or writable host mounts. Review candidate code and pack commands before running.
The fixed benchmark contract supports evaluation integrity; it does not make
arbitrary generated code safe.

## Sharing results

Publish a reviewed finding, minimal example or certificate instead of an entire
run directory. Keep dataset provenance, baseline configuration and any redactions
explicit. Do not label a sanitized or synthetic corpus as the unchanged source
of a historical benchmark result.

Before publishing, inspect every file and the Git history with a secret scanner.
An ignored file can still be tracked if it was committed earlier. A clean current
tree also does not remove a secret from earlier commits, branches, pull requests
or workflow logs.
