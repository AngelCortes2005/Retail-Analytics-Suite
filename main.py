import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime

def read_parquet_file(file_path):
    """Leer el archivo Parquet y devolver un DataFrame."""
    return pd.read_parquet(file_path)

def map_items(items_df, transactions_df):
    """Mapear las columnas del archivo items.csv con el DataFrame de transacciones."""
    merged_df = transactions_df.merge(items_df, on='item_id', how='left')
    return merged_df

def get_top_items(transactions_df):
    """Obtener el top 15 de ítems con mayor cantidad vendida (suma de 'amount') ordenados de mayor a menor."""
    # Agrupar por 'item_id' y sumar los 'amount', luego ordenar de mayor a menor
    top_items = transactions_df.groupby('item_id')['amount'].sum().nlargest(15)
    return transactions_df[transactions_df['item_id'].isin(top_items.index)], top_items.index, top_items

def run_apriori(transactions_df):
    """Ejecutar el algoritmo Apriori y devolver las reglas de asociación."""
    # Crear una tabla de transacciones
    basket = transactions_df.groupby(['id', 'item_id'])['amount'].sum().unstack().reset_index().fillna(0).set_index('id')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0) 

    # Ejecutar el algoritmo Apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)

    # Limitar a 15 reglas
    if len(rules) > 15:
        rules = rules.nlargest(15, 'confidence')  # o puedes cambiar 'confidence' por 'lift' o 'support' si lo prefieres

    return rules

def add_item_names(rules, items_df):
    """Agregar nombres de ítems a las reglas."""
    def get_item_names(item_ids):
        return [items_df.loc[items_df['item_id'] == item, 'item_name'].values[0] for item in item_ids]

    # Aplicar la función para antecedentes y consecuentes
    rules['antecedents_names'] = rules['antecedents'].apply(lambda x: get_item_names(list(x)))
    rules['consequents_names'] = rules['consequents'].apply(lambda x: get_item_names(list(x)))

    return rules

def filter_top_items_rules(rules, top_items):
    """Filtrar reglas para que solo contengan los top items."""
    rules['antecedents_set'] = rules['antecedents'].apply(set)
    rules['consequents_set'] = rules['consequents'].apply(set)

    # Filtrar reglas donde todos los ítems en antecedentes y consecuentes están en el top items
    filtered_rules = rules[rules['antecedents_set'].apply(lambda x: all(item in top_items for item in x)) & 
                           rules['consequents_set'].apply(lambda x: all(item in top_items for item in x))]
    
    return filtered_rules


def main():
    # Cargar los datos
    items_df = pd.read_csv('items.csv')
    transactions_df = read_parquet_file('transactions.parquet')

    # Mapear los ítems
    mapped_df = map_items(items_df, transactions_df)

    # Filtrar por el rango de fechas
    today = pd.Timestamp(datetime.now())
    filtered_df = mapped_df[(mapped_df['date'] >= '2023-01-01') & (mapped_df['date'] <= today)]

    # Obtener el top 15 de ítems junto con las cantidades de ventas
    top_items_df, top_items, item_counts = get_top_items(filtered_df)

    # Unir la información de item_id con item_name
    top_items_with_sales = pd.DataFrame({'item_id': item_counts.index, 'amount': item_counts.values})
    top_items_with_sales = top_items_with_sales.merge(items_df[['item_id', 'item_name']], on='item_id', how='left')

    # Imprimir los 15 artículos más vendidos junto con el amount
    print("Top 15 artículos más vendidos (basado en 'amount'):")
    print(top_items_with_sales[['item_name', 'amount']])

    # Ejecutar Apriori
    rules = run_apriori(top_items_df)

    # Filtrar reglas para que solo contengan los top items
    filtered_rules = filter_top_items_rules(rules, top_items)

    # Agregar nombres de ítems a las reglas
    rules_with_names = add_item_names(filtered_rules, items_df)

    # Formatear la columna de confianza
    rules_with_names['confidence'] = rules_with_names['confidence'].map(lambda x: "{:d}%".format(int(round(x * 100))))

    # Imprimir las sugerencias
    print("\nSugerencias de ítems con métricas:")
    print(rules_with_names[['antecedents_names', 'consequents_names', 'confidence']])

if __name__ == "__main__":
    main()