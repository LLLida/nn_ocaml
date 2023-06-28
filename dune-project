(lang dune 3.7)

(name nn_ocaml)

(generate_opam_files true)

(source
 (github username/reponame))

(authors "Adil Mokhammad")

(maintainers "Maintainer Name")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name nn_ocaml)
 (synopsis "Adil's playground with neural networks")
 (description "With great power comes great responsibility. However, I don't have much power. I think by creating neural networks it's possible to obtain infinite power. What about infinite responsibility then? I don't care.")
 (depends ocaml dune)
 (tags
  (topics "machine learning" your project)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
