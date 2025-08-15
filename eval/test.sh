
codeword=$(python reedmuller/rmencode.py 1 7 00000000)
echo "Codeword: $codeword"
for i in {1..5}
do
    for j in {1..64..2}
    do  
        echo "Number: $j"
        # randomly flip j bits to 1 
        numbers=$(shuf -i 1-128 -n $j)
        new_codeword=$codeword
        for number in $numbers
        do
            # Convert string to array and flip bit at position
            pos=$((number-1))  # Convert to 0-based indexing
            char="${new_codeword:$pos:1}"
            if [ "$char" = "0" ]; then
                new_char="1"
            else
                new_char="0"
            fi
            new_codeword="${new_codeword:0:$pos}$new_char${new_codeword:$((pos+1))}"
        done
        echo "Modified Codeword: $new_codeword"
        decoded=$(python reedmuller/rmdecode.py 1 7 $new_codeword)
        echo "Decoded: $decoded"
        if [ "$decoded" != "00000000" ]; then
            echo "ERROR: Decoded message doesn't match original! Flipped $j bits."
        fi
    done
done