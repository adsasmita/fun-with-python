for f in *
do
echo "${f}/"
echo "${f// /_}/"
mv "${f}/" "${f// /_}/"
done
