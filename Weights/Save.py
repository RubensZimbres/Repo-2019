
Tensorflow:

with tf.Session() as sess:
  sess.run(init)
  save_path = saver.save(sess, "/tmp/saved_model.ckpt")


with tf.Session() as sess:
    saver.restore(sess, "/tmp/saved_model.ckpt")

Keras:

model.save_weights('your_model_weights.h5')
model.load_weights('your_model_weights.h5')

SKLearn:

model=XGBClassifier(max_depth=100, learning_rate=0.7, n_estimators=10, objective='binary:logistic',booster='gbtree',n_jobs=16,eval_metric="error",eval_set=eval_set, verbose=True)
clf=model.fit(x_train,y_train)

from sklearn.externals import joblib
joblib.dump(clf, '/path/your_model.joblib')

model = joblib.load('/path/your_model.joblib')
model.predict(X_train)
