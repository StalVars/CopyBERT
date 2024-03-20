

#SESSION_NAME="xlnet-large-cased-test"
SESSION_NAME=$1

screen -ls "$SESSION_NAME" | (
  IFS=$(printf '\t');
    sed "s/^$IFS//" |
	      while read -r name stuff; do
		            screen -S "$name" -X quit
			      done
			      )
