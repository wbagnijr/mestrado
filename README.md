# Estudo Comparativo de Autoencoders em Imagens: Reconstrução, Geração e Robustez do Espaço Latente (Dissertação de Mestrado)

Este repositório contém as principais estruturas de código (classes e funções) das arquiteturas dos modelos de *Autoencoders* utilizados na dissertação de mestrado intitulada **"Estudo Comparativo de Autoencoders em Imagens: Reconstrução, Geração e Robustez do Espaço Latente"**, apresentada ao Instituto de Computação da UNICAMP.

**Autor:** Wilson Bagni Júnior  
**Orientador:** Prof. Dr. Zanoni Dias  
**Coorientador:** Prof. Dr. Hélio Pedrini

## Resumo do Trabalho

Esta pesquisa realiza um estudo comparativo aprofundado entre diferentes arquiteturas de *Autoencoders* (FCAE, CAE, VAE, AAE e RealNVP-AE), avaliando seus comportamentos em tarefas de reconstrução, geração de dados e organização do espaço latente. Os experimentos foram conduzidos utilizando as bases de dados SVHN e Synthetic Digits (SD).

## Dados da Pesquisa e Modelos Treinados

Além dos códigos disponibilizados neste repositório, os resultados experimentais, métricas de avaliação e modelos treinados obtidos neste estudo estão disponíveis publicamente no **Repositório de Dados de Pesquisa da Unicamp (REDU)**.

* **Acesso aos Dados:** [doi:10.25824/redu/BGNAIH](https://redu.unicamp.br/dataset.xhtml?persistentId=doi:10.25824/redu/BGNAIH)
* **Conteúdo do Depósito:** Arquivos de pesos dos modelos (`.h5`), históricos de treinamento (`.csv`), tabelas de métricas de desempenho (métricas locais, latentes e de parâmetros), índices de erro de reconstrução (RMSE) e classificadores ResNet18 utilizados como avaliadores.
* **Finalidade:** O conjunto de dados foi construído sob diversas dimensões latentes, incluindo testes de robustez cruzada, sendo destinado à reprodutibilidade desta pesquisa de mestrado e para servir como base comparativa para futuros estudos em *Deep Learning* e Visão Computacional.

## Estrutura do Repositório

O código está organizado de forma a refletir a estrutura metodológica da dissertação:

* **`models/`**: Contém as classes das arquiteturas neurais descritas no **Capítulo 2**.
    * `fcae.py` Fully Connected Autoencoder.
    * `cae.py`: Convolutional Autoencoder
    * `vae.py`: Variational Autoencoder.
    * `aae.py`: Adversarial Autoencoder (com discriminador).
    * `realnvpae.py`: Implementação baseada em Fluxos Normalizadores (RealNVP-AE).
    * `latentspace.py`: *Callback* utilizado para monitoramento do espaço latente dos modelos ao longo dos treinamentos.

## Como Citar

Se você utilizar os códigos, modelos ou dados deste repositório em sua pesquisa, por favor, cite a dissertação e o conjunto de dados originais:

### 1. Citação da Dissertação

**Formato ABNT:**
> BAGNI JÚNIOR, Wilson. *Estudo Comparativo de Autoencoders em Imagens: Reconstrução, Geração e Robustez do Espaço Latente*. 2026. Dissertação (Mestrado em Ciência da Computação) - Instituto de Computação, Universidade Estadual de Campinas (UNICAMP), Campinas, 2026. Disponível em: https://repositorio.unicamp.br/.

**Formato BibTeX:**
```
@mastersthesis{bagni_autoencoders_2026,
  author       = {Bagni J\'unior, Wilson},
  title        = {Estudo Comparativo de Autoencoders em Imagens: Reconstru\c{c}\~ao, Gera\c{c}\~ao e Robustez do Espa\c{c}o Latente},
  school       = {Instituto de Computa\c{c}\~ao, Universidade Estadual de Campinas (UNICAMP)},
  year         = {2026},
  address      = {Campinas, SP, Brasil},
  type         = {Disserta\c{c}\~ao de Mestrado},
  url          = {[https://repositorio.unicamp.br/](https://repositorio.unicamp.br/)}
}
```
###  2. Citação do Conjunto de Dados (REDU)

**Formato ABNT:**
> BAGNI JÚNIOR, Wilson. Dados de pesquisa para: Estudo Comparativo de Autoencoders em Imagens: Reconstrução, Geração e Robustez do Espaço Latente. Repositório de Dados de Pesquisa da Unicamp (REDU), 2026. DOI: 10.25824/redu/BGNAIH. Disponível em: https://redu.unicamp.br/dataset.xhtml?persistentId=doi:10.25824/redu/BGNAIH.

**Formato BibTeX:**
```
@misc{bagni_dataset_2026,
  author       = {Bagni J\'unior, Wilson},
  title        = {Dados de pesquisa para: Estudo Comparativo de Autoencoders em Imagens: Reconstru\c{c}\~ao, Gera\c{c}\~ao e Robustez do Espa\c{c}o Latente},
  year         = {2026},
  publisher    = {Reposit\'orio de Dados de Pesquisa da Unicamp (REDU)},
  doi          = {10.25824/redu/BGNAIH},
  url          = {[https://doi.org/10.25824/redu/BGNAIH](https://doi.org/10.25824/redu/BGNAIH)}
}
```
## Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.
Você é livre para utilizar, modificar e distribuir o código deste repositório, inclusive para fins comerciais, desde que a devida atribuição seja dada ao autor original e a nota de direitos autorais seja mantida.
