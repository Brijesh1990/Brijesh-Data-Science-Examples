const mongoose=require("mongoose");
const itemSchema=new mongoose.Schema({
    
    name:{ type: String, required: true },
    email:{ type: String, required: true },
    phone:{ type: String, required: true },
    pin:{ type: String, required: true },
    age:{ type: String, required: true },
    address:{ type: String, required: true }

});

module.exports=mongoose.model('Item',itemSchema);