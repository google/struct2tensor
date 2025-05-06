description: Apache Parquet Dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="expression_impl.parquet" />
<meta itemprop="path" content="Stable" />
</div>

# Module: expression_impl.parquet

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/struct2tensor/blob/master/struct2tensor/expression_impl/parquet.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Apache Parquet Dataset.



#### Example usage:



```
  exp = create_expression_from_parquet_file(filenames)
  docid_project_exp = project.project(exp, [path.Path(["DocId"])])
  pqds = parquet_dataset.calculate_parquet_values([docid_project_exp], exp,
                                                  filenames, batch_size)

  for prensors in pqds:
    doc_id_prensor = prensors[0]
```

## Classes

[`class ParquetDataset`](../expression_impl/parquet/ParquetDataset.md): A dataset which reads columns from a parquet file and returns a prensor.

## Functions

[`calculate_parquet_values(...)`](../expression_impl/parquet/calculate_parquet_values.md): Calculates expressions and returns a parquet dataset.

[`create_expression_from_parquet_file(...)`](../expression_impl/parquet/create_expression_from_parquet_file.md): Creates a placeholder expression from a parquet file.

