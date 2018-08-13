for p1 in $(ls -d */)
do
	#echo $(ls -S ${p1} | head -1)
	c1=`ls -S ${p1} | head -1`
	mv $p1$c1 .
	rm -rf $p1
done


