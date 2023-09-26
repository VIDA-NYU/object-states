
D=/datasets 
#D=/scratch/bs3639

for f in $D/ptgrecipes/pinwheels_*;
    python -m object_states.predict config/pinwheels/pinwheels.yml --file-path $f
done

for f in $D/ptgrecipes/coffee_*;
    python -m object_states.predict config/coffee/coffee.yml --file-path $f
done

for f in $D/ptgrecipes/quesadilla_*;
    python -m object_states.predict config/quesadilla/quesadilla.yml --file-path $f
done

for f in $D/ptgrecipes/oatmeal_*;
    python -m object_states.predict config/oatmeal/oatmeal.yml --file-path $f
done

for f in $D/ptgrecipes/tea_*;
    python -m object_states.predict config/tea/tea.yml --file-path $f
done