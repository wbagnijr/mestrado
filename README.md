# Estudo Comparativo de Autoencoders (Dissertação de Mestrado)

Este repositório contém as principais estruturas de código (classes e funções) das arquiteturas dos modelos de *Autoencoders* utilizados na dissertação de mestrado intitulada **"Estudo Comparativo de Autoencoders"**, apresentada ao Instituto de Computação da UNICAMP.

**Autor:** Wilson Bagni Júnior  
**Orientador:** Prof. Dr. Zanoni Dias  
**Coorientador:** Prof. Dr. Hélio Pedrini

## Resumo do Trabalho

Esta pesquisa realiza um estudo comparativo aprofundado entre diferentes arquiteturas de *Autoencoders* (FCAE, CAE, VAE, AAE e RealNVP-AE), avaliando seus comportamentos em tarefas de reconstrução, geração de dados e organização do espaço latente. Os experimentos foram conduzidos utilizando as bases de dados SVHN e Synthetic Digits (SD).

## Estrutura do Repositório

O código está organizado de forma a refletir a estrutura metodológica da dissertação:

* **`models/`**: Contém as classes das arquiteturas neurais descritas no **Capítulo 2**.
    * `fcae.py` Fully Connected Autoencoder.
    * `cae.py`: Convolutional Autoencoder
    * `vae.py`: Variational Autoencoder.
    * `aae.py`: Adversarial Autoencoder (com discriminador).
    * `realnvpae.py`: Implementação baseada em Fluxos Normalizadores (RealNVP-AE).
    * `latentspace.py`: *Callback* utilizado para monitoramento do espaço latente dos modelos ao longo dos treinamentos.


