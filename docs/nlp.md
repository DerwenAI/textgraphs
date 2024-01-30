The open source `spaCy` library in Python provides full-featured NLP capabilities.
[#honnibal2020spacy](biblio.md#honnibal2020spacy)
This serves as a core component of this project.
Recent releases of `spaCy` have provided features to integrate with selected large models, and also support native features for entity linking.

On the one hand, `spaCy` pipelines offer a broad range of integrations and "opinionated" selections for both utility and ease of use.
The resulting pipelines are optimized for annotating streams of spans of tokens.
On the other hand, the opinionated API calls and the abstractions use for pipeline construction and configuration present some important constraints:

  - Pipelines are not especially well-suited for propagating other forms of generated data, beyond token/span streams.
  - Tokenization used in `spaCy` does not align with the requirements for relation extraction projects of interest.
  - Entity linking capabilities rely on using an internally defined "knowledge base" which is not well-suited for integrating with heterogeneous resources.

Consequently, while `spaCy` serves as a core component for NLP capabilities, this project presents a library of Python class definitions for KG construction which can be extended and configured to accommodate a broad range of LLM components.
These "less opinionated" pipeline definitions, in the broader scope, are optimized for managing streams of KG candidate elements which have been produced by generative AI.
