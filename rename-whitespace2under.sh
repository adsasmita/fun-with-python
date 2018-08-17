for f in *
do
echo "original: ${f}"
echo "renamed to:  ${f//[ ()\[\]]/_}"
mv "${f}" "${f//[ ()\[\]]/_}"
done
