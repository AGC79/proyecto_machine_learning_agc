import data_processing as dp
import training as t
import predictions as pr
import variables as v

rutas_fotos, target = dp.procesar_rutas_fotos(v.base, v.rutas)

dataframe = dp.crear_dataframe_rutas_target(rutas_fotos, target)

lista_fotos_3d = dp.crear_lista_fotos(dataframe, v.img_height, v.img_width)

X_rgb, y_encoded = dp.crear_x_y(lista_fotos_3d, dataframe)

X_tr, X_ts, y_tr, y_ts = dp.split_train_test(X_rgb, y_encoded)

X_tr_nor, X_ts_nor, y_tr_nor, y_ts_nor = dp.normalizar_train_test(X_tr, X_ts, y_tr, y_ts)

X_train_m, X_val_m, y_train_m, y_val_m = dp.split_train_val(X_tr_nor, y_tr_nor, X_ts_nor)

X_train_sfh, y_train_sfh = dp.desordenar_train(X_train_m, y_train_m)

train_gen, val_gen = t.generar_imagenes(X_train_sfh, y_train_sfh, X_val_m, y_val_m)

layers_f = t.crear_capas()

modelo = t.crear_modelo_secuencial(layers_f)

t.compilar_modelo(modelo)

hist_f = t.crear_history(modelo, train_gen, val_gen)

df_hist_f = t.crear_df_history(hist_f)

t.grafico_loss(df_hist_f)

t.grafico_train(df_hist_f)

y_true, y_pred, y_pred_probs = pr.obtener_predicciones(modelo, X_ts_nor, y_ts_nor)

pr.matriz_confusion(y_true, y_pred, v.class_names)

pr.exportar_modelo(modelo)


